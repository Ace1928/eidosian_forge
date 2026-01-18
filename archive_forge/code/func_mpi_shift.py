from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_shift(x, n):
    a, b = x
    return (mpf_shift(a, n), mpf_shift(b, n))