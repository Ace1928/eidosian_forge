from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_cot(x, prec):
    cos, sin = mpi_cos_sin(x, prec + 20)
    return mpi_div(cos, sin, prec)