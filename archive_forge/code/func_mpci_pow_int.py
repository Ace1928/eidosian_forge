from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_pow_int(x, n, prec):
    if n < 0:
        return mpci_div((mpi_one, mpi_zero), mpci_pow_int(x, -n, prec + 20), prec)
    if n == 0:
        return (mpi_one, mpi_zero)
    if n == 1:
        return mpci_pos(x, prec)
    if n == 2:
        return mpci_square(x, prec)
    wp = prec + 20
    result = (mpi_one, mpi_zero)
    while n:
        if n & 1:
            result = mpci_mul(result, x, wp)
            n -= 1
        x = mpci_square(x, wp)
        n >>= 1
    return mpci_pos(result, prec)