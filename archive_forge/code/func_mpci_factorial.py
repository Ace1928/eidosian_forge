from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_factorial(z, prec):
    return mpci_gamma(z, prec, type=1)