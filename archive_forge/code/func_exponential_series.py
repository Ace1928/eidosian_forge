import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def exponential_series(x, prec, type=0):
    """
    Taylor series for cosh/sinh or cos/sin.

    type = 0 -- returns exp(x)  (slightly faster than cosh+sinh)
    type = 1 -- returns (cosh(x), sinh(x))
    type = 2 -- returns (cos(x), sin(x))
    """
    if x < 0:
        x = -x
        sign = 1
    else:
        sign = 0
    r = int(0.5 * prec ** 0.5)
    xmag = bitcount(x) - prec
    r = max(0, xmag + r)
    extra = 10 + 2 * max(r, -xmag)
    wp = prec + extra
    x <<= extra - r
    one = MPZ_ONE << wp
    alt = type == 2
    if prec < EXP_SERIES_U_CUTOFF:
        x2 = a = x * x >> wp
        x4 = x2 * x2 >> wp
        s0 = s1 = MPZ_ZERO
        k = 2
        while a:
            a //= (k - 1) * k
            s0 += a
            k += 2
            a //= (k - 1) * k
            s1 += a
            k += 2
            a = a * x4 >> wp
        s1 = x2 * s1 >> wp
        if alt:
            c = s1 - s0 + one
        else:
            c = s1 + s0 + one
    else:
        u = int(0.3 * prec ** 0.35)
        x2 = a = x * x >> wp
        xpowers = [one, x2]
        for i in xrange(1, u):
            xpowers.append(xpowers[-1] * x2 >> wp)
        sums = [MPZ_ZERO] * u
        k = 2
        while a:
            for i in xrange(u):
                a //= (k - 1) * k
                if alt and k & 2:
                    sums[i] -= a
                else:
                    sums[i] += a
                k += 2
            a = a * xpowers[-1] >> wp
        for i in xrange(1, u):
            sums[i] = sums[i] * xpowers[i] >> wp
        c = sum(sums) + one
    if type == 0:
        s = isqrt_fast(c * c - (one << wp))
        if sign:
            v = c - s
        else:
            v = c + s
        for i in xrange(r):
            v = v * v >> wp
        return v >> extra
    else:
        pshift = wp - 1
        for i in xrange(r):
            c = (c * c >> pshift) - one
        s = isqrt_fast(abs((one << wp) - c * c))
        if sign:
            s = -s
        return (c >> extra, s >> extra)