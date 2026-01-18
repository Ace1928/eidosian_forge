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
@validate_params({'X': ['array-like', 'sparse matrix'], 'Y': ['array-like', 'sparse matrix', None], 'Y_norm_squared': ['array-like', None], 'squared': ['boolean'], 'X_norm_squared': ['array-like', None]}, prefer_skip_nested_validation=True)
def euclidean_distances(X, Y=None, *, Y_norm_squared=None, squared=False, X_norm_squared=None):
    """
    Compute the distance matrix between each pair from a vector array X and Y.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation,
    because this equation potentially suffers from "catastrophic cancellation".
    Also, the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features),             default=None
        An array where each row is a sample and each column is a feature.
        If `None`, method uses `Y=X`.

    Y_norm_squared : array-like of shape (n_samples_Y,) or (n_samples_Y, 1)             or (1, n_samples_Y), default=None
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    squared : bool, default=False
        Return squared Euclidean distances.

    X_norm_squared : array-like of shape (n_samples_X,) or (n_samples_X, 1)             or (1, n_samples_X), default=None
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.

    See Also
    --------
    paired_distances : Distances between pairs of elements of X and Y.

    Notes
    -----
    To achieve a better accuracy, `X_norm_squared`\xa0and `Y_norm_squared` may be
    unused if they are passed as `np.float32`.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    """
    X, Y = check_pairwise_arrays(X, Y)
    if X_norm_squared is not None:
        X_norm_squared = check_array(X_norm_squared, ensure_2d=False)
        original_shape = X_norm_squared.shape
        if X_norm_squared.shape == (X.shape[0],):
            X_norm_squared = X_norm_squared.reshape(-1, 1)
        if X_norm_squared.shape == (1, X.shape[0]):
            X_norm_squared = X_norm_squared.T
        if X_norm_squared.shape != (X.shape[0], 1):
            raise ValueError(f'Incompatible dimensions for X of shape {X.shape} and X_norm_squared of shape {original_shape}.')
    if Y_norm_squared is not None:
        Y_norm_squared = check_array(Y_norm_squared, ensure_2d=False)
        original_shape = Y_norm_squared.shape
        if Y_norm_squared.shape == (Y.shape[0],):
            Y_norm_squared = Y_norm_squared.reshape(1, -1)
        if Y_norm_squared.shape == (Y.shape[0], 1):
            Y_norm_squared = Y_norm_squared.T
        if Y_norm_squared.shape != (1, Y.shape[0]):
            raise ValueError(f'Incompatible dimensions for Y of shape {Y.shape} and Y_norm_squared of shape {original_shape}.')
    return _euclidean_distances(X, Y, X_norm_squared, Y_norm_squared, squared)