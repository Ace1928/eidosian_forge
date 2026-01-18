import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_frac(z, prec, rnd=round_fast):
    a, b = z
    return (mpf_frac(a, prec, rnd), mpf_frac(b, prec, rnd))