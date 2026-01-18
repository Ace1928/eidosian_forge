import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_log_hypot(a, b, prec, rnd):
    """
    Computes log(sqrt(a^2+b^2)) accurately.
    """
    if not b[1]:
        a, b = (b, a)
    if not a[1]:
        if not b[1]:
            if a == b == fzero:
                return fninf
            if fnan in (a, b):
                return fnan
            return finf
        if a == fzero:
            return mpf_log(mpf_abs(b), prec, rnd)
        if a == fnan:
            return fnan
        return finf
    a2 = mpf_mul(a, a)
    b2 = mpf_mul(b, b)
    extra = 20
    h2 = mpf_add(a2, b2, prec + extra)
    cancelled = mpf_add(h2, fnone, 10)
    mag_cancelled = cancelled[2] + cancelled[3]
    if cancelled == fzero or mag_cancelled < -extra // 2:
        h2 = mpf_add(a2, b2, prec + extra - min(a2[2], b2[2]))
    return mpf_shift(mpf_log(h2, prec, rnd), -1)