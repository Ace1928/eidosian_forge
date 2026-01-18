import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def complex_ei_asymptotic(zre, zim, prec):
    _abs = abs
    one = MPZ_ONE << prec
    M = zim * zim + zre * zre >> prec
    xre = tre = (zre << prec) // M
    xim = tim = (-zim << prec) // M
    sre = one + xre
    sim = xim
    k = 2
    while _abs(tre) + _abs(tim) > 1000:
        tre, tim = ((tre * xre - tim * xim) * k >> prec, (tre * xim + tim * xre) * k >> prec)
        sre += tre
        sim += tim
        k += 1
        if k > prec:
            raise NoConvergence
    return (sre, sim)