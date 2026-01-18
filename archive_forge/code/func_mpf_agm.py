import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_agm(a, b, prec, rnd=round_fast):
    """
    Computes the arithmetic-geometric mean agm(a,b) for
    nonnegative mpf values a, b.
    """
    asign, aman, aexp, abc = a
    bsign, bman, bexp, bbc = b
    if asign or bsign:
        raise ComplexResult('agm of a negative number')
    if not (aman and bman):
        if a == fnan or b == fnan:
            return fnan
        if a == finf:
            if b == fzero:
                return fnan
            return finf
        if b == finf:
            if a == fzero:
                return fnan
            return finf
        return fzero
    wp = prec + 20
    amag = aexp + abc
    bmag = bexp + bbc
    mag_delta = amag - bmag
    abs_mag_delta = abs(mag_delta)
    if abs_mag_delta > 10:
        while abs_mag_delta > 10:
            a, b = (mpf_shift(mpf_add(a, b, wp), -1), mpf_sqrt(mpf_mul(a, b, wp), wp))
            abs_mag_delta //= 2
        asign, aman, aexp, abc = a
        bsign, bman, bexp, bbc = b
        amag = aexp + abc
        bmag = bexp + bbc
        mag_delta = amag - bmag
    min_mag = min(amag, bmag)
    max_mag = max(amag, bmag)
    n = 0
    if min_mag < -8:
        n = -min_mag
    elif max_mag > 20:
        n = -max_mag
    if n:
        a = mpf_shift(a, n)
        b = mpf_shift(b, n)
    af = to_fixed(a, wp)
    bf = to_fixed(b, wp)
    g = agm_fixed(af, bf, wp)
    return from_man_exp(g, -wp - n, prec, rnd)