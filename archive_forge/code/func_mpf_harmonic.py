import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpf_harmonic(x, prec, rnd):
    if x in (fzero, fnan, finf):
        return x
    a = mpf_psi0(mpf_add(fone, x, prec + 5), prec)
    return mpf_add(a, mpf_euler(prec + 5, rnd), prec, rnd)