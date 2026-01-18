import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_sin_pi(z, prec, rnd=round_fast):
    a, b = z
    if b == fzero:
        return (mpf_sin_pi(a, prec, rnd), fzero)
    b = mpf_mul(b, mpf_pi(prec + 5), prec + 5)
    if a == fzero:
        return (fzero, mpf_sinh(b, prec, rnd))
    wp = prec + 6
    c, s = mpf_cos_sin_pi(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return (re, im)