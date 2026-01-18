import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpc_zetasum(s, a, n, derivatives, reflect, prec):
    """
    Fast version of mp._zetasum, assuming s = complex, a = integer.
    """
    wp = prec + 10
    derivatives = list(derivatives)
    have_derivatives = derivatives != [0]
    have_one_derivative = len(derivatives) == 1
    sre, sim = s
    critical_line = sre == fhalf
    sre = to_fixed(sre, wp)
    sim = to_fixed(sim, wp)
    if a > 0 and n > ZETASUM_SIEVE_CUTOFF and (not have_derivatives) and (not reflect) and (n < 40000000.0 or sys.maxsize > 2 ** 32):
        re, im = zetasum_sieved(critical_line, sre, sim, a, n, wp)
        xs = [(from_man_exp(re, -wp, prec, 'n'), from_man_exp(im, -wp, prec, 'n'))]
        return (xs, [])
    maxd = max(derivatives)
    if not have_one_derivative:
        derivatives = range(maxd + 1)
    xre = [MPZ_ZERO for d in derivatives]
    xim = [MPZ_ZERO for d in derivatives]
    if reflect:
        yre = [MPZ_ZERO for d in derivatives]
        yim = [MPZ_ZERO for d in derivatives]
    else:
        yre = yim = []
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << 2 * wp
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp - 1)
    wp2 = wp + wp
    for w in xrange(a, a + n + 1):
        log = log_int_fixed(w, wp, ln2)
        cos, sin = cos_sin_fixed(-sim * log >> wp, wp, pi2)
        if critical_line:
            u = one_2wp // isqrt_fast(w << wp2)
        else:
            u = exp_fixed(-sre * log >> wp, wp)
        xterm_re = u * cos >> wp
        xterm_im = u * sin >> wp
        if reflect:
            reciprocal = one_2wp // (u * w)
            yterm_re = reciprocal * cos >> wp
            yterm_im = reciprocal * sin >> wp
        if have_derivatives:
            if have_one_derivative:
                log = pow_fixed(log, maxd, wp)
                xre[0] += xterm_re * log >> wp
                xim[0] += xterm_im * log >> wp
                if reflect:
                    yre[0] += yterm_re * log >> wp
                    yim[0] += yterm_im * log >> wp
            else:
                t = MPZ_ONE << wp
                for d in derivatives:
                    xre[d] += xterm_re * t >> wp
                    xim[d] += xterm_im * t >> wp
                    if reflect:
                        yre[d] += yterm_re * t >> wp
                        yim[d] += yterm_im * t >> wp
                    t = t * log >> wp
        else:
            xre[0] += xterm_re
            xim[0] += xterm_im
            if reflect:
                yre[0] += yterm_re
                yim[0] += yterm_im
    if have_derivatives:
        if have_one_derivative:
            if maxd % 2:
                xre[0] = -xre[0]
                xim[0] = -xim[0]
                if reflect:
                    yre[0] = -yre[0]
                    yim[0] = -yim[0]
        else:
            xre = [(-1) ** d * xre[d] for d in derivatives]
            xim = [(-1) ** d * xim[d] for d in derivatives]
            if reflect:
                yre = [(-1) ** d * yre[d] for d in derivatives]
                yim = [(-1) ** d * yim[d] for d in derivatives]
    xs = [(from_man_exp(xa, -wp, prec, 'n'), from_man_exp(xb, -wp, prec, 'n')) for xa, xb in zip(xre, xim)]
    ys = [(from_man_exp(ya, -wp, prec, 'n'), from_man_exp(yb, -wp, prec, 'n')) for ya, yb in zip(yre, yim)]
    return (xs, ys)