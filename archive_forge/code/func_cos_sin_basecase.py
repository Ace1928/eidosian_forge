import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def cos_sin_basecase(x, prec):
    """
    Compute cos(x), sin(x) as fixed-point numbers, assuming x
    in [0, pi/2). For an arbitrary number, use x' = x - m*(pi/2)
    where m = floor(x/(pi/2)) along with quarter-period symmetries.
    """
    if prec > COS_SIN_CACHE_PREC:
        return exponential_series(x, prec, 2)
    precs = prec - COS_SIN_CACHE_STEP
    t = x >> precs
    n = int(t)
    if n not in cos_sin_cache:
        w = t << 10 + COS_SIN_CACHE_PREC - COS_SIN_CACHE_STEP
        cos_t, sin_t = exponential_series(w, 10 + COS_SIN_CACHE_PREC, 2)
        cos_sin_cache[n] = (cos_t >> 10, sin_t >> 10)
    cos_t, sin_t = cos_sin_cache[n]
    offset = COS_SIN_CACHE_PREC - prec
    cos_t >>= offset
    sin_t >>= offset
    x -= t << precs
    cos = MPZ_ONE << prec
    sin = x
    k = 2
    a = -(x * x >> prec)
    while a:
        a //= k
        cos += a
        k += 1
        a = a * x >> prec
        a //= k
        sin += a
        k += 1
        a = -(a * x >> prec)
    return (cos * cos_t - sin * sin_t >> prec, sin * cos_t + cos * sin_t >> prec)