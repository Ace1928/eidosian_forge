import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_cos(z, prec, rnd=round_fast):
    """Complex cosine. The formula used is cos(a+bi) = cos(a)*cosh(b) -
    sin(a)*sinh(b)*i.

    The same comments apply as for the complex exp: only real
    multiplications are pewrormed, so no cancellation errors are
    possible. The formula is also efficient since we can compute both
    pairs (cos, sin) and (cosh, sinh) in single stwps."""
    a, b = z
    if b == fzero:
        return (mpf_cos(a, prec, rnd), fzero)
    if a == fzero:
        return (mpf_cosh(b, prec, rnd), fzero)
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    re = mpf_mul(c, ch, prec, rnd)
    im = mpf_mul(s, sh, prec, rnd)
    return (re, mpf_neg(im))