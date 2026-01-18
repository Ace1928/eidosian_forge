import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_atan(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return fzero
        if x == finf:
            return atan_inf(0, prec, rnd)
        if x == fninf:
            return atan_inf(1, prec, rnd)
        return fnan
    mag = exp + bc
    if mag > prec + 20:
        return atan_inf(sign, prec, rnd)
    if -mag > prec + 20:
        return mpf_perturb(x, 1 - sign, prec, rnd)
    wp = prec + 30 + abs(mag)
    if mag >= 2:
        x = mpf_rdiv_int(1, x, wp)
        reciprocal = True
    else:
        reciprocal = False
    t = to_fixed(x, wp)
    if sign:
        t = -t
    if wp < ATAN_TAYLOR_PREC:
        a = atan_taylor(t, wp)
    else:
        a = atan_newton(t, wp)
    if reciprocal:
        a = (pi_fixed(wp) >> 1) + 1 - a
    if sign:
        a = -a
    return from_man_exp(a, -wp, prec, rnd)