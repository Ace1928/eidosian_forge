import warnings
import numpy as np
import pandas as pd
from scipy import stats
def nobs_below(row):
    """ B, the number of observations (cen & uncen) below the given
        threshold
        """
    less_than = df[observations] < row['lower_dl']
    less_thanequal = df[observations] <= row['lower_dl']
    uncensored = ~df[censorship]
    censored = df[censorship]
    LTE_censored = df[less_thanequal & censored].shape[0]
    LT_uncensored = df[less_than & uncensored].shape[0]
    return LTE_censored + LT_uncensored