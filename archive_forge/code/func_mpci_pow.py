from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpci_pow(x, y, prec):
    yre, yim = y
    if yim == mpi_zero:
        ya, yb = yre
        if ya == yb:
            sign, man, exp, bc = yb
            if man and exp >= 0:
                return mpci_pow_int(x, (-1) ** sign * int(man << exp), prec)
            if yb == fzero:
                return mpci_pow_int(x, 0, prec)
    wp = prec + 20
    return mpci_exp(mpci_mul(y, mpci_log(x, wp), wp), prec)