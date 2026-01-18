import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def mpf_ge(s, t):
    if s == fnan or t == fnan:
        return False
    return mpf_cmp(s, t) >= 0