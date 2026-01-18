import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_sqrt(s, prec, rnd=round_fast):
    """
    Compute the square root of a nonnegative mpf value. The
    result is correctly rounded.
    """
    sign, man, exp, bc = s
    if sign:
        raise ComplexResult('square root of a negative number')
    if not man:
        return s
    if exp & 1:
        exp -= 1
        man <<= 1
        bc += 1
    elif man == 1:
        return normalize1(sign, man, exp // 2, bc, prec, rnd)
    shift = max(4, 2 * prec - bc + 4)
    shift += shift & 1
    if rnd in 'fd':
        man = isqrt(man << shift)
    else:
        man, rem = sqrtrem(man << shift)
        if rem:
            man = (man << 1) + 1
            shift += 2
    return from_man_exp(man, (exp - shift) // 2, prec, rnd)