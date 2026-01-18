from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def covariance_matrix_grid(self, endog_expval, index):
    from scipy.linalg import toeplitz
    r = np.zeros(len(endog_expval))
    r[0] = 1
    r[1:self.max_lag + 1] = self.dep_params[1:]
    return (toeplitz(r), True)