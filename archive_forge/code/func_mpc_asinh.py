import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_asinh(z, prec, rnd=round_fast):
    a, b = z
    a, b = mpc_asin((b, mpf_neg(a)), prec, rnd)
    return (mpf_neg(b), a)