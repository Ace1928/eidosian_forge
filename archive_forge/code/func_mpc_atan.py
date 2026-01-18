import sys
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, BACKEND
from .libmpf import (\
from .libelefun import (\
def mpc_atan(z, prec, rnd=round_fast):
    a, b = z
    wp = prec + 15
    x = (mpf_add(fone, b, wp), mpf_neg(a))
    y = (mpf_sub(fone, b, wp), a)
    l1 = mpc_log(x, wp)
    l2 = mpc_log(y, wp)
    a, b = mpc_sub(l1, l2, prec, rnd)
    v = (mpf_neg(mpf_shift(b, -1)), mpf_shift(a, -1))
    if v[1] == fnan and mpc_is_inf(z):
        v = (v[0], fzero)
    return v