import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_erf(x, prec, rnd=round_fast):
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            return fzero
        if x == finf:
            return fone
        if x == fninf:
            return fnone
        return fnan
    size = exp + bc
    lg = math.log
    if size > 3 and 2 * (size - 1) + 0.528766 > lg(prec, 2):
        if sign:
            return mpf_perturb(fnone, 0, prec, rnd)
        else:
            return mpf_perturb(fone, 1, prec, rnd)
    if size < -prec:
        x = mpf_shift(x, 1)
        c = mpf_sqrt(mpf_pi(prec + 20), prec + 20)
        return mpf_div(x, c, prec, rnd)
    wp = prec + abs(size) + 25
    t = abs(to_fixed(x, wp))
    t2 = t * t >> wp
    s, term, k = (t, 12345, 1)
    while term:
        t = (t * t2 >> wp) // k
        term = t // (2 * k + 1)
        if k & 1:
            s -= term
        else:
            s += term
        k += 1
    s = (s << wp + 1) // sqrt_fixed(pi_fixed(wp), wp)
    if sign:
        s = -s
    return from_man_exp(s, -wp, prec, rnd)