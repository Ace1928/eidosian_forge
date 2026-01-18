import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_atan2(y, x, prec, rnd=round_fast):
    xsign, xman, xexp, xbc = x
    ysign, yman, yexp, ybc = y
    if not yman:
        if y == fzero and x != fnan:
            if mpf_sign(x) >= 0:
                return fzero
            return mpf_pi(prec, rnd)
        if y in (finf, fninf):
            if x in (finf, fninf):
                return fnan
            if y == finf:
                return mpf_shift(mpf_pi(prec, rnd), -1)
            return mpf_neg(mpf_shift(mpf_pi(prec, negative_rnd[rnd]), -1))
        return fnan
    if ysign:
        return mpf_neg(mpf_atan2(mpf_neg(y), x, prec, negative_rnd[rnd]))
    if not xman:
        if x == fnan:
            return fnan
        if x == finf:
            return fzero
        if x == fninf:
            return mpf_pi(prec, rnd)
        if y == fzero:
            return fzero
        return mpf_shift(mpf_pi(prec, rnd), -1)
    tquo = mpf_atan(mpf_div(y, x, prec + 4), prec + 4)
    if xsign:
        return mpf_add(mpf_pi(prec + 4), tquo, prec, rnd)
    else:
        return mpf_pos(tquo, prec, rnd)