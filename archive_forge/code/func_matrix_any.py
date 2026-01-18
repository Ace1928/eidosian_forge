from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def matrix_any(condition):
    """Check if a condition is true anywhere in a data matrix.

    np.any doesn't handle matrices of type pd.DataFrame

    Parameters
    ----------
    condition : array-like
        Boolean matrix

    Returns
    -------
    any : bool
        True if condition contains any True values, False otherwise
    """
    return np.sum(np.sum(condition)) > 0