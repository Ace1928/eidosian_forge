import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
@constant_memo
def glaisher_fixed(prec):
    wp = prec + 30
    N = int(0.33 * prec + 5)
    ONE = MPZ_ONE << wp
    s = MPZ_ZERO
    for k in range(2, N):
        s += log_int_fixed(k, wp) // k ** 2
    logN = log_int_fixed(N, wp)
    s += (ONE + logN) // N
    s += logN // (N ** 2 * 2)
    pN = N ** 3
    a = 1
    b = -2
    j = 3
    fac = from_int(2)
    k = 1
    while 1:
        D = ((a << wp) + b * logN) // pN
        D = from_man_exp(D, -wp)
        B = mpf_bernoulli(2 * k, wp)
        term = mpf_mul(B, D, wp)
        term = mpf_div(term, fac, wp)
        term = to_fixed(term, wp)
        if abs(term) < 100:
            break
        s -= term
        a, b, pN, j = (b - a * j, -j * b, pN * N, j + 1)
        a, b, pN, j = (b - a * j, -j * b, pN * N, j + 1)
        k += 1
        fac = mpf_mul_int(fac, 2 * k * (2 * k - 1), wp)
    pi = pi_fixed(wp)
    s *= 6
    s = (s << wp) // (pi ** 2 >> wp)
    s += euler_fixed(wp)
    s += to_fixed(mpf_log(from_man_exp(2 * pi, -wp), wp), wp)
    s //= 12
    A = mpf_exp(from_man_exp(s, -wp), wp)
    return to_fixed(A, prec)