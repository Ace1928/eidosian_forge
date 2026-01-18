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
@validate_params({'X': ['array-like', 'sparse matrix'], 'Y': ['array-like', 'sparse matrix']}, prefer_skip_nested_validation=True)
def paired_euclidean_distances(X, Y):
    """Compute the paired euclidean distances between X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input array/matrix X.

    Y : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input array/matrix Y.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Output array/matrix containing the calculated paired euclidean
        distances.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_euclidean_distances
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> paired_euclidean_distances(X, Y)
    array([1., 1.])
    """
    X, Y = check_paired_arrays(X, Y)
    return row_norms(X - Y)