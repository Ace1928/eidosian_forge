import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpc_agm(a, b, prec, rnd=round_fast):
    """
    Complex AGM.

    TODO:
    * check that convergence works as intended
    * optimize
    * select a nonarbitrary branch
    """
    if mpc_is_infnan(a) or mpc_is_infnan(b):
        return (fnan, fnan)
    if mpc_zero in (a, b):
        return (fzero, fzero)
    if mpc_neg(a) == b:
        return (fzero, fzero)
    wp = prec + 20
    eps = mpf_shift(fone, -wp + 10)
    while 1:
        a1 = mpc_shift(mpc_add(a, b, wp), -1)
        b1 = mpc_sqrt(mpc_mul(a, b, wp), wp)
        a, b = (a1, b1)
        size = mpf_min_max([mpc_abs(a, 10), mpc_abs(b, 10)])[1]
        err = mpc_abs(mpc_sub(a, b, 10), 10)
        if size == fzero or mpf_lt(err, mpf_mul(eps, size)):
            return a