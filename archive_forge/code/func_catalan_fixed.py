import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
@constant_memo
def catalan_fixed(prec):
    prec = prec + 20
    a = one = MPZ_ONE << prec
    s, t, n = (0, 1, 1)
    while t:
        a *= 32 * n ** 3 * (2 * n - 1)
        a //= (3 - 16 * n + 16 * n ** 2) ** 2
        t = a * (-1) ** (n - 1) * (40 * n ** 2 - 24 * n + 3) // (n ** 3 * (2 * n - 1))
        s += t
        n += 1
    return s >> 20 + 6