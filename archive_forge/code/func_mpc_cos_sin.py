import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_cos_sin(z, prec, rnd=round_fast):
    a, b = z
    if a == fzero:
        ch, sh = mpf_cosh_sinh(b, prec, rnd)
        return ((ch, fzero), (fzero, sh))
    if b == fzero:
        c, s = mpf_cos_sin(a, prec, rnd)
        return ((c, fzero), (s, fzero))
    wp = prec + 6
    c, s = mpf_cos_sin(a, wp)
    ch, sh = mpf_cosh_sinh(b, wp)
    cre = mpf_mul(c, ch, prec, rnd)
    cim = mpf_mul(s, sh, prec, rnd)
    sre = mpf_mul(s, ch, prec, rnd)
    sim = mpf_mul(c, sh, prec, rnd)
    return ((cre, mpf_neg(cim)), (sre, sim))