import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def stirling_coefficient(n):
    if n in gamma_stirling_cache:
        return gamma_stirling_cache[n]
    p, q = bernfrac(n)
    q *= MPZ(n * (n - 1))
    gamma_stirling_cache[n] = (p, q, bitcount(abs(p)), bitcount(q))
    return gamma_stirling_cache[n]