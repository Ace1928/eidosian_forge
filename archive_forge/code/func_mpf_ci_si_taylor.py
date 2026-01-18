import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpf_ci_si_taylor(x, wp, which=0):
    """
    0 - Ci(x) - (euler+log(x))
    1 - Si(x)
    """
    x = to_fixed(x, wp)
    x2 = -(x * x) >> wp
    if which == 0:
        s, t, k = (0, MPZ_ONE << wp, 2)
    else:
        s, t, k = (x, x, 3)
    while t:
        t = t * x2 // (k * (k - 1)) >> wp
        s += t // k
        k += 2
    return from_man_exp(s, -wp)