import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_acos(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if bc + exp > 0:
        if x not in (fone, fnone):
            raise ComplexResult('acos(x) is real only for -1 <= x <= 1')
        if x == fnone:
            return mpf_pi(prec, rnd)
    wp = prec + 15
    a = mpf_mul(x, x)
    b = mpf_sqrt(mpf_sub(fone, a, wp), wp)
    c = mpf_div(b, mpf_add(fone, x, wp), wp)
    return mpf_shift(mpf_atan(c, prec, rnd), 1)