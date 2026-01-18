import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_mpf_div(p, z, prec, rnd=round_fast):
    """Calculate p/z where p is real efficiently"""
    a, b = z
    m = mpf_add(mpf_mul(a, a), mpf_mul(b, b), prec + 10)
    re = mpf_div(mpf_mul(a, p), m, prec, rnd)
    im = mpf_div(mpf_neg(mpf_mul(b, p)), m, prec, rnd)
    return (re, im)