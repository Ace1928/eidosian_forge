import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpc_ci(z, prec, rnd=round_fast):
    re, im = z
    if im == fzero:
        ci = mpf_ci_si(re, prec, rnd, 0)[0]
        if mpf_sign(re) < 0:
            return (ci, mpf_pi(prec, rnd))
        return (ci, fzero)
    wp = prec + 20
    cre, cim = mpc_ci_si_taylor(re, im, wp, 0)
    cre = mpf_add(cre, mpf_euler(wp), wp)
    ci = mpc_add((cre, cim), mpc_log(z, wp), prec, rnd)
    return ci