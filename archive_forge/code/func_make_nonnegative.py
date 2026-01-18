import warnings
from functools import partial
from numbers import Integral
import numpy as np
from scipy import linalg, sparse
from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array
def make_nonnegative(X, min_value=0):
    """Ensure `X.min()` >= `min_value`.

    Parameters
    ----------
    X : array-like
        The matrix to make non-negative.
    min_value : float, default=0
        The threshold value.

    Returns
    -------
    array-like
        The thresholded array.

    Raises
    ------
    ValueError
        When X is sparse.
    """
    min_ = X.min()
    if min_ < min_value:
        if sparse.issparse(X):
            raise ValueError('Cannot make the data matrix nonnegative because it is sparse. Adding a value to every entry would make it no longer sparse.')
        X = X + (min_value - min_)
    return X