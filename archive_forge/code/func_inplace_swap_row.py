import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def inplace_swap_row(X, m, n):
    """
    Swap two rows of a CSC/CSR matrix in-place.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix whose two rows are to be swapped. It should be of CSR or
        CSC format.

    m : int
        Index of the row of X to be swapped.

    n : int
        Index of the row of X to be swapped.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 3, 3])
    >>> indices = np.array([0, 2, 2])
    >>> data = np.array([8, 2, 5])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 0, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.inplace_swap_row(csr, 0, 1)
    >>> csr.todense()
    matrix([[0, 0, 5],
            [8, 0, 2],
            [0, 0, 0],
            [0, 0, 0]])
    """
    if sp.issparse(X) and X.format == 'csc':
        inplace_swap_row_csc(X, m, n)
    elif sp.issparse(X) and X.format == 'csr':
        inplace_swap_row_csr(X, m, n)
    else:
        _raise_typeerror(X)