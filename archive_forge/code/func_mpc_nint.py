import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_nint(z, prec, rnd=round_fast):
    a, b = z
    return (mpf_nint(a, prec, rnd), mpf_nint(b, prec, rnd))