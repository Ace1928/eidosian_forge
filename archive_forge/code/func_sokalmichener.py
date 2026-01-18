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
def sokalmichener(u, v, w=None):
    """
    Compute the Sokal-Michener dissimilarity between two boolean 1-D arrays.

    The Sokal-Michener dissimilarity between boolean 1-D arrays `u` and `v`,
    is defined as

    .. math::

       \\frac{R}
            {S + R}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k < n`, :math:`R = 2 * (c_{TF} + c_{FT})` and
    :math:`S = c_{FF} + c_{TT}`.

    Parameters
    ----------
    u : (N,) array_like, bool
        Input array.
    v : (N,) array_like, bool
        Input array.
    w : (N,) array_like, optional
        The weights for each value in `u` and `v`. Default is None,
        which gives each value a weight of 1.0

    Returns
    -------
    sokalmichener : double
        The Sokal-Michener dissimilarity between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.sokalmichener([1, 0, 0], [0, 1, 0])
    0.8
    >>> distance.sokalmichener([1, 0, 0], [1, 1, 0])
    0.5
    >>> distance.sokalmichener([1, 0, 0], [2, 0, 0])
    -1.0

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if w is not None:
        w = _validate_weights(w)
    nff, nft, ntf, ntt = _nbool_correspond_all(u, v, w=w)
    return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))