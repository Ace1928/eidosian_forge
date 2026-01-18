import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def inplace_row_scale(X, scale):
    """Inplace row scaling of a CSR or CSC matrix.

    Scale each row of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Matrix to be scaled. It should be of CSR or CSC format.

    scale : ndarray of shape (n_features,), dtype={np.float32, np.float64}
        Array of precomputed sample-wise values to use for scaling.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 2, 3, 4, 5])
    >>> indices = np.array([0, 1, 2, 3, 3])
    >>> data = np.array([8, 1, 2, 5, 6])
    >>> scale = np.array([2, 3, 4, 5])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 5],
            [0, 0, 0, 6]])
    >>> sparsefuncs.inplace_row_scale(csr, scale)
    >>> csr.todense()
     matrix([[16,  2,  0,  0],
             [ 0,  0,  6,  0],
             [ 0,  0,  0, 20],
             [ 0,  0,  0, 30]])
    """
    if sp.issparse(X) and X.format == 'csc':
        inplace_csr_column_scale(X.T, scale)
    elif sp.issparse(X) and X.format == 'csr':
        inplace_csr_row_scale(X, scale)
    else:
        _raise_typeerror(X)