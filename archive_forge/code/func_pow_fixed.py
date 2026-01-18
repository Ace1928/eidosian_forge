import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def pow_fixed(x, n, wp):
    if n == 1:
        return x
    y = MPZ_ONE << wp
    while n:
        if n & 1:
            y = y * x >> wp
            n -= 1
        x = x * x >> wp
        n //= 2
    return y