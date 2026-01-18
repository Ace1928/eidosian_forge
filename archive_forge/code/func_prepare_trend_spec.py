import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def prepare_trend_spec(trend):
    if trend is None or trend == 'n':
        polynomial_trend = np.ones(0)
    elif trend == 'c':
        polynomial_trend = np.r_[1]
    elif trend == 't':
        polynomial_trend = np.r_[0, 1]
    elif trend == 'ct':
        polynomial_trend = np.r_[1, 1]
    elif trend == 'ctt':
        polynomial_trend = np.r_[1, 1, 1]
    else:
        trend = np.array(trend)
        if trend.ndim > 0:
            polynomial_trend = (trend > 0).astype(int)
        else:
            raise ValueError(f"Valid trend inputs are 'c' (constant), 't' (linear trend in time), 'ct' (both), 'ctt' (both with trend squared) or an interable defining a polynomial, e.g., [1, 1, 0, 1] is `a + b*t + ct**3`. Received {trend}")
    k_trend = int(np.sum(polynomial_trend))
    return (polynomial_trend, k_trend)