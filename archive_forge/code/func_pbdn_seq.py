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
def pbdn_seq(n, z):
    """Parabolic cylinder functions Dn(z) and derivatives.

    Parameters
    ----------
    n : int
        Order of the parabolic cylinder function
    z : complex
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of D_i(z), for i=0, ..., i=n.
    dp : ndarray
        Derivatives D_i'(z), for i=0, ..., i=n.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError('arguments must be scalars.')
    if floor(n) != n:
        raise ValueError('n must be an integer.')
    if abs(n) <= 1:
        n1 = 1
    else:
        n1 = n
    cpb, cpd = _specfun.cpbdn(n1, z)
    return (cpb[:n1 + 1], cpd[:n1 + 1])