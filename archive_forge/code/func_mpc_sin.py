import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_sin(z, prec, rnd=round_fast):
    """Complex sine. We have sin(a+bi) = sin(a)*cosh(b) +
    cos(a)*sinh(b)*i. See the docstring for mpc_cos for additional
    comments."""
    a, b = z
    if b == fzero:
        return (mpf_sin(a, prec, rnd), fzero)
    if a == fzero:
        return (fzero, mpf_sinh(b, prec, rnd))
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(s, ch, prec, rnd)
    im = mpf_mul(c, sh, prec, rnd)
    return (re, im)