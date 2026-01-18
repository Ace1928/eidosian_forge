import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
@constant_memo
def pi_fixed(prec, verbose=False, verbose_base=None):
    """
    Compute floor(pi * 2**prec) as a big integer.

    This is done using Chudnovsky's series (see comments in
    libelefun.py for details).
    """
    N = int(prec / 3.3219280948 / 14.181647462 + 2)
    if verbose:
        print('binary splitting with N =', N)
    g, p, q = bs_chudnovsky(0, N, 0, verbose)
    sqrtC = isqrt_fast(CHUD_C << 2 * prec)
    v = p * CHUD_C * sqrtC // ((q + CHUD_A * p) * CHUD_D)
    return v