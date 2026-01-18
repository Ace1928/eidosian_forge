import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_sum(xs, prec=0, rnd=round_fast, absolute=False):
    """
    Sum a list of mpf values efficiently and accurately
    (typically no temporary roundoff occurs). If prec=0,
    the final result will not be rounded either.

    There may be roundoff error or cancellation if extremely
    large exponent differences occur.

    With absolute=True, sums the absolute values.
    """
    man = 0
    exp = 0
    max_extra_prec = prec * 2 or 1000000
    special = None
    for x in xs:
        xsign, xman, xexp, xbc = x
        if xman:
            if xsign and (not absolute):
                xman = -xman
            delta = xexp - exp
            if xexp >= exp:
                if delta > max_extra_prec and (not man or delta - bitcount(abs(man)) > max_extra_prec):
                    man = xman
                    exp = xexp
                else:
                    man += xman << delta
            else:
                delta = -delta
                if delta - xbc > max_extra_prec:
                    if not man:
                        man, exp = (xman, xexp)
                else:
                    man = (man << delta) + xman
                    exp = xexp
        elif xexp:
            if absolute:
                x = mpf_abs(x)
            special = mpf_add(special or fzero, x, 1)
    if special:
        return special
    return from_man_exp(man, exp, prec, rnd)