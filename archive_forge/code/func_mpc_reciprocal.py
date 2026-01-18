import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_reciprocal(z, prec, rnd=round_fast):
    """Calculate 1/z efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a, a), mpf_mul(b, b), prec + 10)
    re = mpf_div(a, m, prec, rnd)
    im = mpf_neg(mpf_div(b, m, prec, rnd))
    return (re, im)