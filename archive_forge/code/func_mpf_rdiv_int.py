import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_rdiv_int(n, t, prec, rnd=round_fast):
    """Floating-point division n/t with a Python integer as numerator"""
    sign, man, exp, bc = t
    if not n or not man:
        return mpf_div(from_int(n), t, prec, rnd)
    if n < 0:
        sign ^= 1
        n = -n
    extra = prec + bc + 5
    quot, rem = divmod(n << extra, man)
    if rem:
        quot = (quot << 1) + 1
        extra += 1
        return normalize1(sign, quot, -exp - extra, bitcount(quot), prec, rnd)
    return normalize(sign, quot, -exp - extra, bitcount(quot), prec, rnd)