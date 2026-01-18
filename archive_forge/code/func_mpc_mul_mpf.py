import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_mul_mpf(z, p, prec, rnd=round_fast):
    a, b = z
    re = mpf_mul(a, p, prec, rnd)
    im = mpf_mul(b, p, prec, rnd)
    return (re, im)