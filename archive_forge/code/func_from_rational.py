import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def from_rational(p, q, prec, rnd=round_fast):
    """Create a raw mpf from a rational number p/q, round if
    necessary."""
    return mpf_div(from_int(p), from_int(q), prec, rnd)