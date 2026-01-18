from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_atan2(y, x, prec):
    ya, yb = y
    xa, xb = x
    if ya == yb == fzero:
        if mpf_ge(xa, fzero):
            return mpi_zero
        return mpi_pi(prec)
    if mpf_ge(xa, fzero):
        if mpf_ge(ya, fzero):
            a = mpf_atan2(ya, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xa, prec, round_floor)
        if mpf_ge(yb, fzero):
            b = mpf_atan2(yb, xa, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    elif mpf_ge(ya, fzero):
        b = mpf_atan2(ya, xa, prec, round_ceiling)
        if mpf_le(xb, fzero):
            a = mpf_atan2(yb, xb, prec, round_floor)
        else:
            a = mpf_atan2(ya, xb, prec, round_floor)
    elif mpf_le(yb, fzero):
        a = mpf_atan2(yb, xa, prec, round_floor)
        if mpf_le(xb, fzero):
            b = mpf_atan2(ya, xb, prec, round_ceiling)
        else:
            b = mpf_atan2(yb, xb, prec, round_ceiling)
    else:
        b = mpf_pi(prec, round_ceiling)
        a = mpf_neg(b)
    return (a, b)