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
def haversine_distances(X, Y=None):
    """Compute the Haversine distance between samples in X and Y.

    The Haversine (or great circle) distance is the angular distance between
    two points on the surface of a sphere. The first coordinate of each point
    is assumed to be the latitude, the second is the longitude, given
    in radians. The dimension of the data must be 2.

    .. math::
       D(x, y) = 2\\arcsin[\\sqrt{\\sin^2((x_{lat} - y_{lat}) / 2)
                                + \\cos(x_{lat})\\cos(y_{lat})\\
                                sin^2((x_{lon} - y_{lon}) / 2)}]

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, 2)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, 2), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        The distance matrix.

    Notes
    -----
    As the Earth is nearly spherical, the haversine formula provides a good
    approximation of the distance between two points of the Earth surface, with
    a less than 1% error on average.

    Examples
    --------
    We want to calculate the distance between the Ezeiza Airport
    (Buenos Aires, Argentina) and the Charles de Gaulle Airport (Paris,
    France).

    >>> from sklearn.metrics.pairwise import haversine_distances
    >>> from math import radians
    >>> bsas = [-34.83333, -58.5166646]
    >>> paris = [49.0083899664, 2.53844117956]
    >>> bsas_in_radians = [radians(_) for _ in bsas]
    >>> paris_in_radians = [radians(_) for _ in paris]
    >>> result = haversine_distances([bsas_in_radians, paris_in_radians])
    >>> result * 6371000/1000  # multiply by Earth radius to get kilometers
    array([[    0.        , 11099.54035582],
           [11099.54035582,     0.        ]])
    """
    from ..metrics import DistanceMetric
    return DistanceMetric.get_metric('haversine').pairwise(X, Y)