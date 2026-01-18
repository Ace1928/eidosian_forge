import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def exp_basecase(x, prec):
    """
    Compute exp(x) as a fixed-point number. Works for any x,
    but for speed should have |x| < 1. For an arbitrary number,
    use exp(x) = exp(x-m*log(2)) * 2^m where m = floor(x/log(2)).
    """
    if prec > EXP_COSH_CUTOFF:
        return exponential_series(x, prec, 0)
    r = int(prec ** 0.5)
    prec += r
    s0 = s1 = MPZ_ONE << prec
    k = 2
    a = x2 = x * x >> prec
    while a:
        a //= k
        s0 += a
        k += 1
        a //= k
        s1 += a
        k += 1
        a = a * x2 >> prec
    s1 = s1 * x >> prec
    s = s0 + s1
    u = r
    while r:
        s = s * s >> prec
        r -= 1
    return s >> u