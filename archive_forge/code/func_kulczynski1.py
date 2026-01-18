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
def kulczynski1(u, v, *, w=None):
    """
    Compute the Kulczynski 1 dissimilarity between two boolean 1-D arrays.

    The Kulczynski 1 dissimilarity between two boolean 1-D arrays `u` and `v`
    of length ``n``, is defined as

    .. math::

         \\frac{c_{11}}
              {c_{01} + c_{10}}

    where :math:`c_{ij}` is the number of occurrences of
    :math:`\\mathtt{u[k]} = i` and :math:`\\mathtt{v[k]} = j` for
    :math:`k \\in {0, 1, ..., n-1}`.

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
    kulczynski1 : float
        The Kulczynski 1 distance between vectors `u` and `v`.

    Notes
    -----
    This measure has a minimum value of 0 and no upper limit.
    It is un-defined when there are no non-matches.

    .. versionadded:: 1.8.0

    References
    ----------
    .. [1] Kulczynski S. et al. Bulletin
           International de l'Academie Polonaise des Sciences
           et des Lettres, Classe des Sciences Mathematiques
           et Naturelles, Serie B (Sciences Naturelles). 1927;
           Supplement II: 57-203.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.kulczynski1([1, 0, 0], [0, 1, 0])
    0.0
    >>> distance.kulczynski1([True, False, False], [True, True, False])
    1.0
    >>> distance.kulczynski1([True, False, False], [True])
    0.5
    >>> distance.kulczynski1([1, 0, 0], [3, 1, 0])
    -3.0

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    if w is not None:
        w = _validate_weights(w)
    _, nft, ntf, ntt = _nbool_correspond_all(u, v, w=w)
    return ntt / (ntf + nft)