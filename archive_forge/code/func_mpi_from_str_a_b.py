from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_from_str_a_b(x, y, percent, prec):
    wp = prec + 20
    xa = from_str(x, wp, round_floor)
    xb = from_str(x, wp, round_ceiling)
    y = from_str(y, wp, round_ceiling)
    assert mpf_ge(y, fzero)
    if percent:
        y = mpf_mul(MAX(mpf_abs(xa), mpf_abs(xb)), y, wp, round_ceiling)
        y = mpf_div(y, from_int(100), wp, round_ceiling)
    a = mpf_sub(xa, y, prec, round_floor)
    b = mpf_add(xb, y, prec, round_ceiling)
    return (a, b)