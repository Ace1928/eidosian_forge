import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_expjpi(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin_pi(re, prec, rnd)
    sign, man, exp, bc = im
    wp = prec + 10
    if man:
        wp += max(0, exp + bc)
    im = mpf_neg(mpf_mul(mpf_pi(wp), im, wp))
    if re == fzero:
        return (mpf_exp(im, prec, rnd), fzero)
    ey = mpf_exp(im, prec + 10)
    c, s = mpf_cos_sin_pi(re, prec + 10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return (re, im)