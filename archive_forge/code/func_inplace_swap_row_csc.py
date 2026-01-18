import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def inplace_swap_row_csc(X, m, n):
    """Swap two rows of a CSC matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two rows are to be swapped. It should be of
        CSC format.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.
    """
    for t in [m, n]:
        if isinstance(t, np.ndarray):
            raise TypeError('m and n should be valid integers')
    if m < 0:
        m += X.shape[0]
    if n < 0:
        n += X.shape[0]
    m_mask = X.indices == m
    X.indices[X.indices == n] = m
    X.indices[m_mask] = n