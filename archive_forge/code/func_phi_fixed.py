import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
@constant_memo
def phi_fixed(prec):
    """
    Computes the golden ratio, (1+sqrt(5))/2
    """
    prec += 10
    a = isqrt_fast(MPZ_FIVE << 2 * prec) + (MPZ_ONE << prec)
    return a >> 11