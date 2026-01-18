import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_min_max(seq):
    min = max = seq[0]
    for x in seq[1:]:
        if mpf_lt(x, min):
            min = x
        if mpf_gt(x, max):
            max = x
    return (min, max)