from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable
def lb4(x):
    s, p = acorr_ljungbox(x, lags=4, return_df=True)
    return (s[-1], p[-1])