import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def agm_fixed(a, b, prec):
    """
    Fixed-point computation of agm(a,b), assuming
    a, b both close to unit magnitude.
    """
    i = 0
    while 1:
        anew = a + b >> 1
        if i > 4 and abs(a - anew) < 8:
            return a
        b = isqrt_fast(a * b)
        a = anew
        i += 1
    return a