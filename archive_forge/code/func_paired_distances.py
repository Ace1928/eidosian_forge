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
@validate_params({'X': ['array-like'], 'Y': ['array-like'], 'metric': [StrOptions(set(PAIRED_DISTANCES)), callable]}, prefer_skip_nested_validation=True)
def paired_distances(X, Y, *, metric='euclidean', **kwds):
    """
    Compute the paired distances between X and Y.

    Compute the distances between (X[0], Y[0]), (X[1], Y[1]), etc...

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Array 1 for distance computation.

    Y : ndarray of shape (n_samples, n_features)
        Array 2 for distance computation.

    metric : str or callable, default="euclidean"
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        specified in PAIRED_DISTANCES, including "euclidean",
        "manhattan", or "cosine".
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from `X` as input and return a value indicating
        the distance between them.

    **kwds : dict
        Unused parameters.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.

    See Also
    --------
    sklearn.metrics.pairwise_distances : Computes the distance between every pair of
        samples.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_distances
    >>> X = [[0, 1], [1, 1]]
    >>> Y = [[0, 1], [2, 1]]
    >>> paired_distances(X, Y)
    array([0., 1.])
    """
    if metric in PAIRED_DISTANCES:
        func = PAIRED_DISTANCES[metric]
        return func(X, Y)
    elif callable(metric):
        X, Y = check_paired_arrays(X, Y)
        distances = np.zeros(len(X))
        for i in range(len(X)):
            distances[i] = metric(X[i], Y[i])
        return distances