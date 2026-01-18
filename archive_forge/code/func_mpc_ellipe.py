import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpc_ellipe(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        if re == finf:
            return (fzero, finf)
        if mpf_le(re, fone):
            return (mpf_ellipe(re, prec, rnd), fzero)
    wp = prec + 15
    mag = mpc_abs(z, 1)
    p = max(mag[2] + mag[3], 0) - wp
    h = mpf_shift(fone, p)
    K = mpc_ellipk(z, 2 * wp)
    Kh = mpc_ellipk(mpc_add_mpf(z, h, 2 * wp), 2 * wp)
    Kdiff = mpc_shift(mpc_sub(Kh, K, wp), -p)
    t = mpc_sub(mpc_one, z, wp)
    b = mpc_mul(Kdiff, mpc_shift(z, 1), wp)
    return mpc_mul(t, mpc_add(K, b, wp), prec, rnd)