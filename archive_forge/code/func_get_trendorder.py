from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like
import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.tsa.tsatools as tsa
def get_trendorder(trend='c'):
    if trend == 'c':
        trendorder = 1
    elif trend in ('n', 'nc'):
        trendorder = 0
    elif trend == 'ct':
        trendorder = 2
    elif trend == 'ctt':
        trendorder = 3
    else:
        raise ValueError(f'Unkown trend: {trend}')
    return trendorder