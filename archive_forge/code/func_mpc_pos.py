import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_pos(z, prec, rnd=round_fast):
    a, b = z
    return (mpf_pos(a, prec, rnd), mpf_pos(b, prec, rnd))