import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def mean_variance_axis(X, axis, weights=None, return_sum_weights=False):
    """Compute mean and variance along an axis on a CSR or CSC matrix.

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It can be of CSR or CSC format.

    axis : {0, 1}
        Axis along which the axis should be computed.

    weights : ndarray of shape (n_samples,) or (n_features,), default=None
        If axis is set to 0 shape is (n_samples,) or
        if axis is set to 1 shape is (n_features,).
        If it is set to None, then samples are equally weighted.

        .. versionadded:: 0.24

    return_sum_weights : bool, default=False
        If True, returns the sum of weights seen for each feature
        if `axis=0` or each sample if `axis=1`.

        .. versionadded:: 0.24

    Returns
    -------

    means : ndarray of shape (n_features,), dtype=floating
        Feature-wise means.

    variances : ndarray of shape (n_features,), dtype=floating
        Feature-wise variances.

    sum_weights : ndarray of shape (n_features,), dtype=floating
        Returned if `return_sum_weights` is `True`.

    Examples
    --------
    >>> from sklearn.utils import sparsefuncs
    >>> from scipy import sparse
    >>> import numpy as np
    >>> indptr = np.array([0, 3, 4, 4, 4])
    >>> indices = np.array([0, 1, 2, 2])
    >>> data = np.array([8, 1, 2, 5])
    >>> scale = np.array([2, 3, 2])
    >>> csr = sparse.csr_matrix((data, indices, indptr))
    >>> csr.todense()
    matrix([[8, 1, 2],
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0]])
    >>> sparsefuncs.mean_variance_axis(csr, axis=0)
    (array([2.  , 0.25, 1.75]), array([12.    ,  0.1875,  4.1875]))
    """
    _raise_error_wrong_axis(axis)
    if sp.issparse(X) and X.format == 'csr':
        if axis == 0:
            return _csr_mean_var_axis0(X, weights=weights, return_sum_weights=return_sum_weights)
        else:
            return _csc_mean_var_axis0(X.T, weights=weights, return_sum_weights=return_sum_weights)
    elif sp.issparse(X) and X.format == 'csc':
        if axis == 0:
            return _csc_mean_var_axis0(X, weights=weights, return_sum_weights=return_sum_weights)
        else:
            return _csr_mean_var_axis0(X.T, weights=weights, return_sum_weights=return_sum_weights)
    else:
        _raise_typeerror(X)