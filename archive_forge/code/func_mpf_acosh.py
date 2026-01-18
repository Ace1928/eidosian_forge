import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_acosh(x, prec, rnd=round_fast):
    wp = prec + 15
    if mpf_cmp(x, fone) == -1:
        raise ComplexResult('acosh(x) is real only for x >= 1')
    q = mpf_sqrt(mpf_add(mpf_mul(x, x), fnone, wp), wp)
    return mpf_log(mpf_add(x, q, wp), prec, rnd)