import itertools
import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance
from .. import config_context
from ..exceptions import DataConversionWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import (
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import parse_version, sp_base_version
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _num_samples, check_non_negative
from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
@validate_params({'X': ['array-like', 'sparse matrix'], 'Y': ['array-like', 'sparse matrix', None], 'gamma': [Interval(Real, 0, None, closed='neither'), Hidden(np.ndarray), None]}, prefer_skip_nested_validation=True)
def laplacian_kernel(X, Y=None, gamma=None):
    """Compute the laplacian kernel between X and Y.

    The laplacian kernel is defined as::

        K(x, y) = exp(-gamma ||x-y||_1)

    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <laplacian_kernel>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    gamma : float, default=None
        If None, defaults to 1.0 / n_features. Otherwise it should be strictly positive.

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        The kernel matrix.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import laplacian_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> laplacian_kernel(X, Y)
    array([[0.71..., 0.51...],
           [0.51..., 0.71...]])
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = -gamma * manhattan_distances(X, Y)
    np.exp(K, K)
    return K