import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_div(z, w, prec, rnd=round_fast):
    a, b = z
    c, d = w
    wp = prec + 10
    mag = mpf_add(mpf_mul(c, c), mpf_mul(d, d), wp)
    t = mpf_add(mpf_mul(a, c), mpf_mul(b, d), wp)
    u = mpf_sub(mpf_mul(b, c), mpf_mul(a, d), wp)
    return (mpf_div(t, mag, prec, rnd), mpf_div(u, mag, prec, rnd))