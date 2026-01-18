import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpc_harmonic(z, prec, rnd):
    if z[1] == fzero:
        return (mpf_harmonic(z[0], prec, rnd), fzero)
    a = mpc_psi0(mpc_add_mpf(z, fone, prec + 5), prec)
    return mpc_add_mpf(a, mpf_euler(prec + 5, rnd), prec, rnd)