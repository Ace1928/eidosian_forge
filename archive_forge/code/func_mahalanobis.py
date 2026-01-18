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
def mahalanobis(u, v, VI):
    """
    Compute the Mahalanobis distance between two 1-D arrays.

    The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

    .. math::

       \\sqrt{ (u-v) V^{-1} (u-v)^T }

    where ``V`` is the covariance matrix.  Note that the argument `VI`
    is the inverse of ``V``.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    VI : array_like
        The inverse of the covariance matrix.

    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    >>> distance.mahalanobis([1, 0, 0], [0, 1, 0], iv)
    1.0
    >>> distance.mahalanobis([0, 2, 0], [0, 1, 0], iv)
    1.0
    >>> distance.mahalanobis([2, 0, 0], [0, 1, 0], iv)
    1.7320508075688772

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)