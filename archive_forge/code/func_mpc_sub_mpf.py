import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_sub_mpf(z, p, prec=0, rnd=round_fast):
    a, b = z
    return (mpf_sub(a, p, prec, rnd), b)