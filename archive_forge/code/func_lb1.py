from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable
def lb1(x):
    s, p = acorr_ljungbox(x, lags=1, return_df=True)
    return (s[0], p[0])