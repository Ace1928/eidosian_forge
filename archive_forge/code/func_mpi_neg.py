from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_neg(s, prec=0):
    sa, sb = s
    a = mpf_neg(sb, prec, round_floor)
    b = mpf_neg(sa, prec, round_ceiling)
    return (a, b)