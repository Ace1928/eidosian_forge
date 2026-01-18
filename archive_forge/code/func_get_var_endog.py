from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like
import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.tsa.tsatools as tsa
def get_var_endog(y, lags, trend='c', has_constant='skip'):
    """
    Make predictor matrix for VAR(p) process

    Z := (Z_0, ..., Z_T).T (T x Kp)
    Z_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

    Ref: Lütkepohl p.70 (transposed)

    has_constant can be 'raise', 'add', or 'skip'. See add_constant.
    """
    nobs = len(y)
    Z = np.array([y[t - lags:t][::-1].ravel() for t in range(lags, nobs)])
    if trend != 'n':
        Z = tsa.add_trend(Z, prepend=True, trend=trend, has_constant=has_constant)
    return Z