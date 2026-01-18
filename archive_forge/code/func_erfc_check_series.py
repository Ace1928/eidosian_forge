import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def erfc_check_series(x, prec):
    n = to_int(x)
    if n ** 2 * 1.44 > prec:
        return True
    return False