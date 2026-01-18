import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def gamma_taylor_coefficients(inprec):
    """
    Gives the Taylor coefficients of 1/gamma(1+x) as
    a list of fixed-point numbers. Enough coefficients are returned
    to ensure that the series converges to the given precision
    when x is in [0.5, 1.5].
    """
    if inprec < 400:
        prec = inprec + (10 - inprec % 10)
    elif inprec < 1000:
        prec = inprec + (30 - inprec % 30)
    else:
        prec = inprec
    if prec in gamma_taylor_cache:
        return (gamma_taylor_cache[prec], prec)
    if prec < 1000:
        N = int(prec ** 0.76 + 2)
    else:
        N = int(prec ** 0.787 + 2)
    for cprec in gamma_taylor_cache:
        if cprec > prec:
            coeffs = [x >> cprec - prec for x in gamma_taylor_cache[cprec][-N:]]
            if inprec < 1000:
                gamma_taylor_cache[prec] = coeffs
            return (coeffs, prec)
    if prec > 1000:
        prec = int(prec * 1.2)
    wp = prec + 20
    A = [0] * N
    A[0] = MPZ_ZERO
    A[1] = MPZ_ONE << wp
    A[2] = euler_fixed(wp)
    zeta_values = zeta_array(N, wp)
    for k in xrange(3, N):
        a = -A[2] * A[k - 1] >> wp
        for j in xrange(2, k):
            a += (-1) ** j * zeta_values[j] * A[k - j] >> wp
        a //= 1 - k
        A[k] = a
    A = [a >> 20 for a in A]
    A = A[::-1]
    A = A[:-1]
    gamma_taylor_cache[prec] = A
    return gamma_taylor_coefficients(inprec)