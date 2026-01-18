import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def from_npfloat(x, prec=113, rnd=round_fast):
    """Create a raw mpf from a numpy float, rounding if necessary.
    If prec >= 113, the result is guaranteed to represent exactly the
    same number as the input. If prec is not specified, use prec=113."""
    y = float(x)
    if x == y:
        return from_float(y, prec, rnd)
    import numpy as np
    if np.isfinite(x):
        m, e = np.frexp(x)
        return from_man_exp(int(np.ldexp(m, 113)), int(e - 113), prec, rnd)
    if np.isposinf(x):
        return finf
    if np.isneginf(x):
        return fninf
    return fnan