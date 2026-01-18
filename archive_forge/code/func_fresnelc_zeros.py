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
def fresnelc_zeros(nt):
    """Compute nt complex zeros of cosine Fresnel integral C(z).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    fresnelc_zeros: ndarray
        Zeros of the cosine Fresnel integral

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if floor(nt) != nt or nt <= 0 or (not isscalar(nt)):
        raise ValueError('Argument must be positive scalar integer.')
    return _specfun.fcszo(1, nt)