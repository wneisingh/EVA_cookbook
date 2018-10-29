

def func_POT(Q,Q_Threshold,n_neighbours=5,use_cumecs=True,use_k_range=True,verbose=False):
    """ Peaks-Over-Threshold analysis
    Inputs:
    -------
    Q : pandas Series
        Discharges over time [m3/day; daily time step]
    Q_Threshold : float
                  Theshold over which to consider peaks [m3/day]
    n_neighbours : float
                   Number of days before and after a given day to consider when identifying peaks
    use_cumecs : boolean
                 Whether to convert the inputs to m3/s before performing analysis
    use_k_range : boolean
                  If True, several k values will be tried for each distribution and the best-performing respective results are stored.
                  If False, a few selected k values for each distribution (based on Goda 2010) are tried.
                  While True allows more k values to be tried, False allows the implementation of more sophisticated distribution selection/rejection tests that are only available for selected k values. 
    verbose : boolean
              Whether or not to print things during execution


    Returns:
    --------
    output : namedtuple
             Fields:
             - DF_POT: pandas DataFrame
                       The original timeseries with an added boolean column indicated the selected peaks
                       Each row is a Q observation
             - DF_Peaks: pandas DataFrame
                         For each identified peak, the rank and respective y and F values for the best-fitting parameter value combination of each distribution.
                         For the best-perfoming distribution among all, also a column with the return periods estimated by this function for each peak
                         Each row is a Q peak detected
             - dfABr: pandas DataFrame
                      For each distribution, the best-performing parameters A, B (and k if applicable) and the corresponding correlation coefficient r
                      Each row is a distribution type; distributions are ordered from best to worst-performing
             - DF_R: pandas DataFrame
                     Return periods (0.1 year to 500 years by default) and corresponding estimated discharges by the best-performing function (x_u)
                     Each row is a return period
             - y_F: lambda function
                    Inverse of the best-performing distribution: for a given F, returns the corresponding reduced variable (y)
             - F_xAB: lambda function
                      Best-performing function: for a given x (discharge) and shape parameters A and B (and k if applicable), returns the corresponding cdf value
                      #TODO: Since we've already estimated the best-performing A,B,k combination, this function could be just a function of x

    """
    
    from collections import namedtuple
    import numpy as np
    import pandas as pd
    from scipy import stats
    

    DF_POT = Q.rename('Q').to_frame()

    if use_cumecs:
        DF_POT = DF_POT / (24*3600)
        Q_Threshold = Q_Threshold / (24*3600)



    DF_POT.loc[:,'PeakFlag'] = False  # Pre-allocate peak flag column
    for row in DF_POT.itertuples():
        today = row.Index
        # If today's Q is larger than the maximum in the previous n_neighbours days and coming n_neighbours days
        if row.Q > max(DF_POT.loc[today-pd.to_timedelta(f'{n_neighbours}D'):today-pd.to_timedelta('1D'),'Q'].max(),  # Max in the previous days
                       DF_POT.loc[today+pd.to_timedelta('1D'):today+pd.to_timedelta(f'{n_neighbours}D'),'Q'].max()): # Max in the coming days
            DF_POT.at[today,'PeakFlag'] = True  # Flag it
    # Create a DF that contains only the peaks-over-threshold
    DF_Peaks = DF_POT.loc[(DF_POT.PeakFlag)&(DF_POT.Q>Q_Threshold),'Q'].copy(deep=True).rename('Peaks').to_frame()
    # Get number of peak events
    N_T = DF_Peaks.shape[0]
    # Get (approximate) period length in years
    K = (Q.index.max() - Q.index.min()).days/365
    Lambda = N_T / K

    # 1 - Define candidate functions
    dictFunctions = {'FT-I': {'y_m': lambda F_m : -np.log(-np.log(F_m)),
                              'F' : lambda x,A,B : np.exp(-np.exp(-(x-B)/A)),
                              'alpha': 0.44,
                              'beta': 0.12
                              },
                     'FT-II': {'y_m': lambda F_m, k : k * ((-np.log(F_m))**(-1/k) - 1),
                               'F' : lambda x,A,B,k : np.exp(-(1+(x-B)/(k*A))**(-k)),
                               'k': np.array([2.5, 3.33, 5.0, 10.0]),  #TODO: Instead of pre-defining values we could test several within the loop and stick to the best-performing one 
                               'alpha': lambda k : 0.44 + 0.62/k,
                               'beta': lambda k : 0.12 + 0.11/k
                               },
                     'Weibull': {'y_m': lambda F_m, k : (-np.log(1-F_m))**(1/k),
                                 'F' : lambda x,A,B,k : 1 - np.exp(-((x-B)/A)**k),
                                 'k': np.array([0.75, 1.0, 1.4, 2.0]),
                                 'alpha': lambda k : 0.20 + 0.27/np.sqrt(k),
                                 'beta': lambda k : 0.20 + 0.23/np.sqrt(k)
                                 }
                     }

    # Assign k ranges to test several values
    for function in dictFunctions.keys():
        if 'k' in dictFunctions[function].keys():
            dictFunctions[function]['k_range'] = np.linspace(dictFunctions[function]['k'].min(),
                                                             dictFunctions[function]['k'].max(),
                                                             num=30
                                                             )

    # Pre-allocate DataFrame to store parameters A and B and correlation coefficient
    dfABr = pd.DataFrame(columns=['A','B','k','r'])  # Index will be created as we iterate through the functions and values

    #2 Prepare ordered statistics of extreme data
    # sort in descending order + assign ranks
    DF_Peaks.loc[:,'Ranks'] = DF_Peaks.loc[:,'Peaks'].rank(ascending=False)
    DF_Peaks = DF_Peaks.sort_values(by='Ranks')

    for function in dictFunctions.keys():  # For each candidate function:
        if verbose: print(f"Going through {function}...",end='')
        # If this distribution type has a k parameter
        if 'k' in dictFunctions[function].keys():
            if use_k_range: # Option A: use all k values in range
                k_values = dictFunctions[function]['k_range']
            else: # Option B: use specific defined k values
                k_values = dictFunctions[function]['k']
                
            for k in k_values:
                if verbose: print(f"Trying {function}, k = {k:.2f}...",end='')
                Fm = 1 - ((DF_Peaks.loc[:,'Ranks']-dictFunctions[function]['alpha'](k)) 
                         /(N_T+dictFunctions[function]['beta'](k))
                          )
                ym = dictFunctions[function]['y_m'](Fm,k)
                params = stats.linregress(x=ym,  # Independent variable observations (what we feed the function)
                                          y=DF_Peaks.loc[:,'Peaks']  # Dependent variable observations (what we want to reproduce)
                                          )
                
                #TODO: If use_k_range==False --> REC (p. 397) can be implemented to reject distribution
                
                if ((function not in dfABr.index)  # If the function doesn't have A,B,k and r values stored (i.e. if this is the first run for this function)
                    or 
                    (params.rvalue >= dfABr.at[function,'r'])  # The r computed for this k is higher than the one currently stored
                    ):
                    # Get respective return periods:
                    R = 1.0/(Lambda*(1-dictFunctions[function]['F'](DF_Peaks.loc[:,'Peaks'],  # = x_u
                                                                  params.slope,  # A
                                                                  params.intercept,  # B
                                                                  k)))
                    # Store/replace the values for this function with the ones from the current iteration
                    dfABr.loc[function] = [params.slope,  # A
                                           params.intercept,  # B
                                           k,
                                           params.rvalue]
                    # Assign the Fm and ym values for this k to be representative of the function in DF_Peaks
                    DF_Peaks.loc[:,f'{function}_Fm'] = Fm
                    DF_Peaks.loc[:,f'{function}_ym'] = ym
                    DF_Peaks.loc[:,f'{function}_R'] = R
                if verbose: print("done")
                    
        else: # If there is no such thing as a k for this distribution: same things but not a function of k
            # Plotting positions
            DF_Peaks.loc[:,f'{function}_Fm'] = 1 - ((DF_Peaks.loc[:,'Ranks']-dictFunctions[function]['alpha'])
                                                   /(N_T+dictFunctions[function]['beta'])
                                                    )
            DF_Peaks.loc[:,f'{function}_ym'] = dictFunctions[function]['y_m'](DF_Peaks.loc[:,f'{function}_Fm'])
            ### Fit linear function to the data and retrieve the parameters
            params = stats.linregress(x=DF_Peaks.loc[:,f'{function}_ym'],  # Independent variable observations (what we feed the function)
                                      y=DF_Peaks.loc[:,'Peaks']  # Dependent variable observations (what we want to reproduce)
                                      )
            R = 1/(Lambda*(1-dictFunctions[function]['F'](DF_Peaks.loc[:,'Peaks'],  # = x_u
                                                          params.slope,  # A
                                                          params.intercept,  # B
                                                          )))
            DF_Peaks.loc[:,f'{function}_R'] = R
            dfABr.loc[f'{function}'] = [params.slope,  # A
                                        params.intercept,  # B
                                        None,  # k (doesn't apply)
                                        params.rvalue]  # r

        if verbose: print(" done")
        
    DF_Peaks.iloc[:,2:] = DF_Peaks.iloc[:,2:].sort_index(axis='columns')  # Sort columns but keep Peaks and Ranks in place

    dfABr = dfABr.sort_values(by='r',ascending=False)

    DF_R = pd.DataFrame(data={'R': np.concatenate([np.arange(start=0.1,stop=1,step=0.1),  # Smaller interval for <= 1 year
                                               np.arange(start=1,stop=501,step=1)]),  # Larger interval for >= 1 year
                          'F': None,
                          'y_R': None,  # Reduced variate for a given R
                          'x_u': None
                          })

    # R -> F
    # The probability of exceedance on a given year is by definition the inverse of the return period
    # DF_R.loc[:,'F'] = 1/DF_R.loc[:,'R']  # For annual maxima (I think)
    DF_R.loc[:,'F'] = 1 - 1/(Lambda*DF_R.loc[:,'R'])

    # F -> y_R
    # For a given probability of exceedance we have the 
    DF_R.loc[:,'y_R'] = dictFunctions[dfABr.iloc[0].name]['y_m'](DF_R.loc[:,'F'],k)  # Eq. 11.16

    # y_R -> x_u
    DF_R.loc[:,'x_u'] = DF_R.loc[:,'y_R'] * dfABr.iloc[0].A + dfABr.iloc[0].B  # Eq. 11.23

    # CREATE NAMEDTUPLE FOR OUTPUT
    # Create class
    Output = namedtuple('Output',
                        ['DF_POT','DF_Peaks','dfABr','DF_R','y_F','F_xAB'])
    # Create object of class
    output = Output(DF_POT,
                    DF_Peaks,
                    dfABr,
                    DF_R,
                    dictFunctions[dfABr.iloc[0].name]['y_m'],
                    dictFunctions[dfABr.iloc[0].name]['F']
                    )

    # RETURN
    return output


