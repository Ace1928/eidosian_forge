import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def lqmn(m, n, z):
    """Sequence of associated Legendre functions of the second kind.

    Computes the associated Legendre function of the second kind of order m and
    degree n, ``Qmn(z)`` = :math:`Q_n^m(z)`, and its derivative, ``Qmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Qmn(z)`` and
    ``Qmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : complex
        Input value.

    Returns
    -------
    Qmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Qmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(m) or m < 0:
        raise ValueError('m must be a non-negative integer.')
    if not isscalar(n) or n < 0:
        raise ValueError('n must be a non-negative integer.')
    if not isscalar(z):
        raise ValueError('z must be scalar.')
    m = int(m)
    n = int(n)
    mm = max(1, m)
    nn = max(1, n)
    if iscomplex(z):
        q, qd = _specfun.clqmn(mm, nn, z)
    else:
        q, qd = _specfun.lqmn(mm, nn, z)
    return (q[:m + 1, :n + 1], qd[:m + 1, :n + 1])