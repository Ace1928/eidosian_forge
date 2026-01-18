import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
@constant_memo
def e_fixed(prec):
    """
    Computes exp(1). This is done using the ordinary Taylor series for
    exp, with binary splitting. For a description of the algorithm,
    see:

        http://numbers.computation.free.fr/Constants/
            Algorithms/splitting.html
    """
    N = int(1.1 * prec / math.log(prec) + 20)
    p, q = bspe(0, N)
    return (p + q << prec) // q