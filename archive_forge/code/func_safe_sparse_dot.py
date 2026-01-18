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
def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.

    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from sklearn.utils.extmath import safe_sparse_dot
    >>> X = csr_matrix([[1, 2], [3, 4], [5, 6]])
    >>> dot_product = safe_sparse_dot(X, X.T)
    >>> dot_product.toarray()
    array([[ 5, 11, 17],
           [11, 25, 39],
           [17, 39, 61]])
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b
    if sparse.issparse(a) and sparse.issparse(b) and dense_output and hasattr(ret, 'toarray'):
        return ret.toarray()
    return ret