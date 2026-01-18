import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_ellipe(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return mpf_shift(mpf_pi(prec, rnd), -1)
        if x == fninf:
            return finf
        if x == fnan:
            return x
        if x == finf:
            raise ComplexResult
    if x == fone:
        return fone
    wp = prec + 20
    mag = exp + bc
    if mag < -wp:
        return mpf_shift(mpf_pi(prec, rnd), -1)
    p = max(mag, 0) - wp
    h = mpf_shift(fone, p)
    K = mpf_ellipk(x, 2 * wp)
    Kh = mpf_ellipk(mpf_sub(x, h), 2 * wp)
    Kdiff = mpf_shift(mpf_sub(K, Kh), -p)
    t = mpf_sub(fone, x)
    b = mpf_mul(Kdiff, mpf_shift(x, 1), wp)
    return mpf_mul(t, mpf_add(K, b), prec, rnd)