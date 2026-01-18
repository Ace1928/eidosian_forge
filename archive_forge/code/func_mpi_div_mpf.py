from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_div_mpf(s, t, prec):
    return mpi_div(s, (t, t), prec)