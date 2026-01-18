import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def from_pickable(x):
    sign, man, exp, bc = x
    return (sign, MPZ(man, 16), exp, bc)