from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_lt(s, t):
    sa, sb = s
    ta, tb = t
    if mpf_lt(sb, ta):
        return True
    if mpf_ge(sa, tb):
        return False
    return None