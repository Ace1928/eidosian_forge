import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def incr_mean_variance_axis(X, *, axis, last_mean, last_var, last_n, weights=None):
    """Compute incremental mean and variance along an axis on a CSR or CSC matrix.

    last_mean, last_var are the statistics computed at the last step by this
    function. Both must be initialized to 0-arrays of the proper size, i.e.
    the number of features in X. last_n is the number of samples encountered
    until now.

    Parameters
    ----------
    X : CSR or CSC sparse matrix of shape (n_samples, n_features)
        Input data.

    axis : {0, 1}
        Axis along which the axis should be computed.

    last_mean : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Array of means to update with the new data X.
        Should be of shape (n_features,) if axis=0 or (n_samples,) if axis=1.

    last_var : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Array of variances to update with the new data X.
        Should be of shape (n_features,) if axis=0 or (n_samples,) if axis=1.

    last_n : float or ndarray of shape (n_features,) or (n_samples,),             dtype=floating
        Sum of the weights seen so far, excluding the current weights
        If not float, it should be of shape (n_features,) if
        axis=0 or (n_samples,) if axis=1. If float it corresponds to
        having same weights for all samples (or features).

    weights : ndarray of shape (n_samples,) or (n_features,), default=None
        If axis is set to 0 shape is (n_samples,) or
        if axis is set to 1 shape is (n_features,).
        If it is set to None, then samples are equally weighted.

        .. versionadded:: 0.24

    Returns
    -------
    means : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Updated feature-wise means if axis = 0 or
        sample-wise means if axis = 1.

    variances : ndarray of shape (n_features,) or (n_samples,), dtype=floating
        Updated feature-wise variances if axis = 0 or
        sample-wise variances if axis = 1.

    n : ndarray of shape (n_features,) or (n_samples,), dtype=integral
        Updated number of seen samples per feature if axis=0
        or number of seen features per sample if axis=1.

        If weights is not None, n is a sum of the weights of the seen
        samples or features instead of the actual number of seen
        samples or features.

    Notes
    -----
    NaNs are ignored in the algorithm.

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
    >>> sparsefuncs.incr_mean_variance_axis(
    ...     csr, axis=0, last_mean=np.zeros(3), last_var=np.zeros(3), last_n=2
    ... )
    (array([1.3..., 0.1..., 1.1...]), array([8.8..., 0.1..., 3.4...]),
    array([6., 6., 6.]))
    """
    _raise_error_wrong_axis(axis)
    if not (sp.issparse(X) and X.format in ('csc', 'csr')):
        _raise_typeerror(X)
    if np.size(last_n) == 1:
        last_n = np.full(last_mean.shape, last_n, dtype=last_mean.dtype)
    if not np.size(last_mean) == np.size(last_var) == np.size(last_n):
        raise ValueError('last_mean, last_var, last_n do not have the same shapes.')
    if axis == 1:
        if np.size(last_mean) != X.shape[0]:
            raise ValueError(f'If axis=1, then last_mean, last_n, last_var should be of size n_samples {X.shape[0]} (Got {np.size(last_mean)}).')
    elif np.size(last_mean) != X.shape[1]:
        raise ValueError(f'If axis=0, then last_mean, last_n, last_var should be of size n_features {X.shape[1]} (Got {np.size(last_mean)}).')
    X = X.T if axis == 1 else X
    if weights is not None:
        weights = _check_sample_weight(weights, X, dtype=X.dtype)
    return _incr_mean_var_axis0(X, last_mean=last_mean, last_var=last_var, last_n=last_n, weights=weights)