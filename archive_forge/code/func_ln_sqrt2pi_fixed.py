import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
@constant_memo
def ln_sqrt2pi_fixed(prec):
    wp = prec + 10
    return to_fixed(mpf_log(mpf_shift(mpf_pi(wp), 1), wp), prec - 1)