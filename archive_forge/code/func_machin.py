import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def machin(coefs, prec, hyperbolic=False):
    """
    Evaluate a Machin-like formula, i.e., a linear combination of
    acot(n) or acoth(n) for specific integer values of n, using fixed-
    point arithmetic. The input should be a list [(c, n), ...], giving
    c*acot[h](n) + ...
    """
    extraprec = 10
    s = MPZ_ZERO
    for a, b in coefs:
        s += MPZ(a) * acot_fixed(MPZ(b), prec + extraprec, hyperbolic)
    return s >> extraprec