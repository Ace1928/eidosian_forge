import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def seuclidean(u, v, V):
    """
    Return the standardized Euclidean distance between two 1-D arrays.

    The standardized Euclidean distance between two n-vectors `u` and `v` is

    .. math::

       \\sqrt{\\sum\\limits_i \\frac{1}{V_i} \\left(u_i-v_i \\right)^2}

    ``V`` is the variance vector; ``V[I]`` is the variance computed over all the i-th
    components of the points. If not passed, it is automatically computed.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    V : (N,) array_like
        `V` is an 1-D array of component variances. It is usually computed
        among a larger collection vectors.

    Returns
    -------
    seuclidean : double
        The standardized Euclidean distance between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.seuclidean([1, 0, 0], [0, 1, 0], [0.1, 0.1, 0.1])
    4.4721359549995796
    >>> distance.seuclidean([1, 0, 0], [0, 1, 0], [1, 0.1, 0.1])
    3.3166247903553998
    >>> distance.seuclidean([1, 0, 0], [0, 1, 0], [10, 0.1, 0.1])
    3.1780497164141406

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    V = _validate_vector(V, dtype=np.float64)
    if V.shape[0] != u.shape[0] or u.shape[0] != v.shape[0]:
        raise TypeError('V must be a 1-D array of the same dimension as u and v.')
    return euclidean(u, v, w=1 / V)