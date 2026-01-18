import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def round_int(x, n, rnd):
    if rnd == round_nearest:
        if x >= 0:
            t = x >> n - 1
            if t & 1 and (t & 2 or x & h_mask[n < 300][n]):
                return (t >> 1) + 1
            else:
                return t >> 1
        else:
            return -round_int(-x, n, rnd)
    if rnd == round_floor:
        return x >> n
    if rnd == round_ceiling:
        return -(-x >> n)
    if rnd == round_down:
        if x >= 0:
            return x >> n
        return -(-x >> n)
    if rnd == round_up:
        if x >= 0:
            return -(-x >> n)
        return x >> n