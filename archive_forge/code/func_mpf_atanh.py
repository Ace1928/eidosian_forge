import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_atanh(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man and exp:
        if x in (fzero, fnan):
            return x
        raise ComplexResult('atanh(x) is real only for -1 <= x <= 1')
    mag = bc + exp
    if mag > 0:
        if mag == 1 and man == 1:
            return [finf, fninf][sign]
        raise ComplexResult('atanh(x) is real only for -1 <= x <= 1')
    wp = prec + 15
    if mag < -8:
        if mag < -wp:
            return mpf_perturb(x, sign, prec, rnd)
        wp += -mag
    a = mpf_add(x, fone, wp)
    b = mpf_sub(fone, x, wp)
    return mpf_shift(mpf_log(mpf_div(a, b, wp), prec, rnd), -1)