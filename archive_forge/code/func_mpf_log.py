import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_log(x, prec, rnd=round_fast):
    """
    Compute the natural logarithm of the mpf value x. If x is negative,
    ComplexResult is raised.
    """
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return fninf
        if x == finf:
            return finf
        if x == fnan:
            return fnan
    if sign:
        raise ComplexResult('logarithm of a negative number')
    wp = prec + 20
    if man == 1:
        if not exp:
            return fzero
        return from_man_exp(exp * ln2_fixed(wp), -wp, prec, rnd)
    mag = exp + bc
    abs_mag = abs(mag)
    if abs_mag <= 1:
        tsign = 1 - abs_mag
        if tsign:
            tman = (MPZ_ONE << bc) - man
        else:
            tman = man - (MPZ_ONE << bc - 1)
        tbc = bitcount(tman)
        cancellation = bc - tbc
        if cancellation > wp:
            t = normalize(tsign, tman, abs_mag - bc, tbc, tbc, 'n')
            return mpf_perturb(t, tsign, prec, rnd)
        else:
            wp += cancellation
    if abs_mag > 10000:
        if bitcount(abs_mag) > wp:
            return from_man_exp(exp * ln2_fixed(wp), -wp, prec, rnd)
    if wp <= LOG_TAYLOR_PREC:
        m = log_taylor_cached(lshift(man, wp - bc), wp)
        if mag:
            m += mag * ln2_fixed(wp)
    else:
        optimal_mag = -wp // LOG_AGM_MAG_PREC_RATIO
        n = optimal_mag - mag
        x = mpf_shift(x, n)
        wp += -optimal_mag
        m = -log_agm(to_fixed(x, wp), wp)
        m -= n * ln2_fixed(wp)
    return from_man_exp(m, -wp, prec, rnd)