from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_neg(x, prec=0):
    a, b = x
    return (mpi_neg(a, prec), mpi_neg(b, prec))