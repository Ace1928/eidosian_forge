import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_pos(s, prec=0, rnd=round_fast):
    """Calculate 0+s for a raw mpf (i.e., just round s to the specified
    precision)."""
    if prec:
        sign, man, exp, bc = s
        if not man and exp:
            return s
        return normalize1(sign, man, exp, bc, prec, rnd)
    return s