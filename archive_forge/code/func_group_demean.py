from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def group_demean(self, x, use_bincount=True):
    nobs = float(len(x))
    means_g = group_sums(x / nobs, self.group_int, use_bincount=use_bincount)
    x_demeaned = x - means_g[self.group_int]
    return (x_demeaned, means_g)