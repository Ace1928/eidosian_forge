import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def bs_chudnovsky(a, b, level, verbose):
    """
    Computes the sum from a to b of the series in the Chudnovsky
    formula. Returns g, p, q where p/q is the sum as an exact
    fraction and g is a temporary value used to save work
    for recursive calls.
    """
    if b - a == 1:
        g = MPZ((6 * b - 5) * (2 * b - 1) * (6 * b - 1))
        p = b ** 3 * CHUD_C ** 3 // 24
        q = (-1) ** b * g * (CHUD_A + CHUD_B * b)
    else:
        if verbose and level < 4:
            print('  binary splitting', a, b)
        mid = (a + b) // 2
        g1, p1, q1 = bs_chudnovsky(a, mid, level + 1, verbose)
        g2, p2, q2 = bs_chudnovsky(mid, b, level + 1, verbose)
        p = p1 * p2
        g = g1 * g2
        q = q1 * p2 + q2 * g1
    return (g, p, q)