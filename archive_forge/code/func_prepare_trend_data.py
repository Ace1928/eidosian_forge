import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def prepare_trend_data(polynomial_trend, k_trend, nobs, offset=1):
    time_trend = np.arange(offset, nobs + offset)
    trend_data = np.zeros((nobs, k_trend))
    i = 0
    for k in polynomial_trend.nonzero()[0]:
        if k == 0:
            trend_data[:, i] = np.ones(nobs)
        else:
            trend_data[:, i] = time_trend ** k
        i += 1
    return trend_data