import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def mpc_ci_si_taylor(re, im, wp, which=0):
    if re[1]:
        mag = re[2] + re[3]
    elif im[1]:
        mag = im[2] + im[3]
    if im[1]:
        mag = max(mag, im[2] + im[3])
    if mag > 2 or mag < -wp:
        raise NotImplementedError
    wp += 2 - mag
    zre = to_fixed(re, wp)
    zim = to_fixed(im, wp)
    z2re = zim * zim - zre * zre >> wp
    z2im = -2 * zre * zim >> wp
    tre = zre
    tim = zim
    one = MPZ_ONE << wp
    if which == 0:
        sre, sim, tre, tim, k = (0, 0, MPZ_ONE << wp, 0, 2)
    else:
        sre, sim, tre, tim, k = (zre, zim, zre, zim, 3)
    while max(abs(tre), abs(tim)) > 2:
        f = k * (k - 1)
        tre, tim = ((tre * z2re - tim * z2im) // f >> wp, (tre * z2im + tim * z2re) // f >> wp)
        sre += tre // k
        sim += tim // k
        k += 2
    return (from_man_exp(sre, -wp), from_man_exp(sim, -wp))