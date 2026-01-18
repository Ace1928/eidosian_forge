import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def mpf_cosh(x, prec, rnd=round_fast):
    return mpf_cosh_sinh(x, prec, rnd)[0]