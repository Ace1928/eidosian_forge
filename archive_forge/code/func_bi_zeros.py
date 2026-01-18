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
def bi_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Bi and its derivative.

    Computes the first `nt` zeros, b, of the Airy function Bi(x);
    first `nt` zeros, b', of the derivative of the Airy function Bi'(x);
    the corresponding values Bi(b');
    and the corresponding values Bi'(b).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    b : ndarray
        First `nt` zeros of Bi(x)
    bp : ndarray
        First `nt` zeros of Bi'(x)
    bi : ndarray
        Values of Bi(x) evaluated at first `nt` zeros of Bi'(x)
    bip : ndarray
        Values of Bi'(x) evaluated at first `nt` zeros of Bi(x)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy import special
    >>> b, bp, bi, bip = special.bi_zeros(3)
    >>> b
    array([-1.17371322, -3.2710933 , -4.83073784])
    >>> bp
    array([-2.29443968, -4.07315509, -5.51239573])
    >>> bi
    array([-0.45494438,  0.39652284, -0.36796916])
    >>> bip
    array([ 0.60195789, -0.76031014,  0.83699101])

    """
    kf = 2
    if not isscalar(nt) or floor(nt) != nt or nt <= 0:
        raise ValueError('nt must be a positive integer scalar.')
    return _specfun.airyzo(nt, kf)