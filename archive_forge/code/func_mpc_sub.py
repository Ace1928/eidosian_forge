import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_sub(z, w, prec=0, rnd=round_fast):
    a, b = z
    c, d = w
    return (mpf_sub(a, c, prec, rnd), mpf_sub(b, d, prec, rnd))