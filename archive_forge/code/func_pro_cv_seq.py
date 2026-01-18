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
def pro_cv_seq(m, n, c):
    """Characteristic values for prolate spheroidal wave functions.

    Compute a sequence of characteristic values for the prolate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError('Arguments must be scalars.')
    if n != floor(n) or m != floor(m):
        raise ValueError('Modes must be integers.')
    if n - m > 199:
        raise ValueError('Difference between n and m is too large.')
    maxL = n - m + 1
    return _specfun.segv(m, n, c, 1)[1][:maxL]