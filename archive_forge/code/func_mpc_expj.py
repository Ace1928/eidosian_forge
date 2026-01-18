import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_expj(z, prec, rnd='f'):
    re, im = z
    if im == fzero:
        return mpf_cos_sin(re, prec, rnd)
    if re == fzero:
        return (mpf_exp(mpf_neg(im), prec, rnd), fzero)
    ey = mpf_exp(mpf_neg(im), prec + 10)
    c, s = mpf_cos_sin(re, prec + 10)
    re = mpf_mul(ey, c, prec, rnd)
    im = mpf_mul(ey, s, prec, rnd)
    return (re, im)