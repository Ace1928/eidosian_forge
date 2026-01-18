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
@validate_params({'X': ['array-like', 'sparse matrix'], 'Y': ['array-like', 'sparse matrix', None]}, prefer_skip_nested_validation=True)
def manhattan_distances(X, Y=None):
    """Compute the L1 distances between the vectors in X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An array where each row is a sample and each column is a feature.
        If `None`, method uses `Y=X`.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Pairwise L1 distances.

    Notes
    -----
    When X and/or Y are CSR sparse matrices and they are not already
    in canonical format, this function modifies them in-place to
    make them canonical.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances([[3]], [[3]])
    array([[0.]])
    >>> manhattan_distances([[3]], [[2]])
    array([[1.]])
    >>> manhattan_distances([[2]], [[3]])
    array([[1.]])
    >>> manhattan_distances([[1, 2], [3, 4]],         [[1, 2], [0, 3]])
    array([[0., 2.],
           [4., 4.]])
    """
    X, Y = check_pairwise_arrays(X, Y)
    if issparse(X) or issparse(Y):
        X = csr_matrix(X, copy=False)
        Y = csr_matrix(Y, copy=False)
        X.sum_duplicates()
        Y.sum_duplicates()
        D = np.zeros((X.shape[0], Y.shape[0]))
        _sparse_manhattan(X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, D)
        return D
    return distance.cdist(X, Y, 'cityblock')