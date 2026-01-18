import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_hypot(x, y, prec, rnd=round_fast):
    """Compute the Euclidean norm sqrt(x**2 + y**2) of two raw mpfs
    x and y."""
    if y == fzero:
        return mpf_abs(x, prec, rnd)
    if x == fzero:
        return mpf_abs(y, prec, rnd)
    hypot2 = mpf_add(mpf_mul(x, x), mpf_mul(y, y), prec + 4)
    return mpf_sqrt(hypot2, prec, rnd)