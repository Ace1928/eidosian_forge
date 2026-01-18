from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_mid(s, prec):
    sa, sb = s
    return mpf_shift(mpf_add(sa, sb, prec, round_nearest), -1)