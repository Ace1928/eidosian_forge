from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_add(s, t, prec=0):
    sa, sb = s
    ta, tb = t
    a = mpf_add(sa, ta, prec, round_floor)
    b = mpf_add(sb, tb, prec, round_ceiling)
    if a == fnan:
        a = fninf
    if b == fnan:
        b = finf
    return (a, b)