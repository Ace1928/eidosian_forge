import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def inplace_swap_row_csr(X, m, n):
    """Swap two rows of a CSR matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two rows are to be swapped. It should be of
        CSR format.

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
    if m > n:
        m, n = (n, m)
    indptr = X.indptr
    m_start = indptr[m]
    m_stop = indptr[m + 1]
    n_start = indptr[n]
    n_stop = indptr[n + 1]
    nz_m = m_stop - m_start
    nz_n = n_stop - n_start
    if nz_m != nz_n:
        X.indptr[m + 2:n] += nz_n - nz_m
        X.indptr[m + 1] = m_start + nz_n
        X.indptr[n] = n_stop - nz_m
    X.indices = np.concatenate([X.indices[:m_start], X.indices[n_start:n_stop], X.indices[m_stop:n_start], X.indices[m_start:m_stop], X.indices[n_stop:]])
    X.data = np.concatenate([X.data[:m_start], X.data[n_start:n_stop], X.data[m_stop:n_start], X.data[m_start:m_stop], X.data[n_stop:]])