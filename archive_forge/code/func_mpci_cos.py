from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_cos(x, prec):
    a, b = x
    wp = prec + 10
    c, s = mpi_cos_sin(a, wp)
    ch, sh = mpi_cosh_sinh(b, wp)
    re = mpi_mul(c, ch, prec)
    im = mpi_mul(s, sh, prec)
    return (re, mpi_neg(im))