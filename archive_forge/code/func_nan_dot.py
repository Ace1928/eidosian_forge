import numpy as np
import pandas as pd
import scipy.linalg
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import array_like
def nan_dot(A, B):
    """
    Returns np.dot(left_matrix, right_matrix) with the convention that
    nan * 0 = 0 and nan * x = nan if x != 0.

    Parameters
    ----------
    A, B : ndarray
    """
    should_be_nan_1 = np.dot(np.isnan(A), B != 0)
    should_be_nan_2 = np.dot(A != 0, np.isnan(B))
    should_be_nan = should_be_nan_1 + should_be_nan_2
    C = np.dot(np.nan_to_num(A), np.nan_to_num(B))
    C[should_be_nan] = np.nan
    return C