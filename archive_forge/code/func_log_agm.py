import math
from bisect import bisect
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE, BACKEND
from .libmpf import (
from .libintmath import ifib
def log_agm(x, prec):
    """
    Fixed-point computation of -log(x) = log(1/x), suitable
    for large precision. It is required that 0 < x < 1. The
    algorithm used is the Sasaki-Kanada formula

        -log(x) = pi/agm(theta2(x)^2,theta3(x)^2). [1]

    For faster convergence in the theta functions, x should
    be chosen closer to 0.

    Guard bits must be added by the caller.

    HYPOTHESIS: if x = 2^(-n), n bits need to be added to
    account for the truncation to a fixed-point number,
    and this is the only significant cancellation error.

    The number of bits lost to roundoff is small and can be
    considered constant.

    [1] Richard P. Brent, "Fast Algorithms for High-Precision
        Computation of Elementary Functions (extended abstract)",
        http://wwwmaths.anu.edu.au/~brent/pd/RNC7-Brent.pdf

    """
    x2 = x * x >> prec
    s = a = b = x2
    while a:
        b = b * x2 >> prec
        a = a * b >> prec
        s += a
    s += MPZ_ONE << prec
    s = s * s >> prec - 2
    s = s * isqrt_fast(x << prec) >> prec
    t = a = b = x
    while a:
        b = b * x2 >> prec
        a = a * b >> prec
        t += a
    t = (MPZ_ONE << prec) + (t << 1)
    t = t * t >> prec
    p = agm_fixed(s, t, prec)
    return (pi_fixed(prec) << prec) // p