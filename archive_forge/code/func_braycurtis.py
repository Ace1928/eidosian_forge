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
def braycurtis(u, v, w=None):
    """
    Compute the Bray-Curtis distance between two 1-D arrays.

    Bray-Curtis distance is defined as

    .. math::

       \\sum{|u_i-v_i|} / \\sum{|u_i+v_i|}

    The Bray-Curtis distance is in the range [0, 1] if all coordinates are
    positive, and is undefined if the inputs are of length zero.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    braycurtis : double
        The Bray-Curtis distance between 1-D arrays `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.braycurtis([1, 0, 0], [0, 1, 0])
    1.0
    >>> distance.braycurtis([1, 1, 0], [0, 1, 0])
    0.33333333333333331

    """
    u = _validate_vector(u)
    v = _validate_vector(v, dtype=np.float64)
    l1_diff = abs(u - v)
    l1_sum = abs(u + v)
    if w is not None:
        w = _validate_weights(w)
        l1_diff = w * l1_diff
        l1_sum = w * l1_sum
    return l1_diff.sum() / l1_sum.sum()