import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpf_gamma(x, prec, rnd='d', type=0):
    """
    This function implements multipurpose evaluation of the gamma
    function, G(x), as well as the following versions of the same:

    type = 0 -- G(x)                    [standard gamma function]
    type = 1 -- G(x+1) = x*G(x+1) = x!  [factorial]
    type = 2 -- 1/G(x)                  [reciprocal gamma function]
    type = 3 -- log(|G(x)|)             [log-gamma function, real part]
    """
    sign, man, exp, bc = x
    if not man:
        if x == fzero:
            if type == 1:
                return fone
            if type == 2:
                return fzero
            raise ValueError('gamma function pole')
        if x == finf:
            if type == 2:
                return fzero
            return finf
        return fnan
    if type == 3:
        wp = prec + 20
        if exp + bc > wp and (not sign):
            return mpf_sub(mpf_mul(x, mpf_log(x, wp), wp), x, prec, rnd)
    is_integer = exp >= 0
    if is_integer:
        if sign:
            if type == 2:
                return fzero
            raise ValueError('gamma function pole')
        n = man << exp
        if n < SMALL_FACTORIAL_CACHE_SIZE:
            if type == 0:
                return mpf_pos(small_factorial_cache[n - 1], prec, rnd)
            if type == 1:
                return mpf_pos(small_factorial_cache[n], prec, rnd)
            if type == 2:
                return mpf_div(fone, small_factorial_cache[n - 1], prec, rnd)
            if type == 3:
                return mpf_log(small_factorial_cache[n - 1], prec, rnd)
    else:
        n = int(man >> -exp)
    mag = exp + bc
    gamma_size = n * mag
    if type == 3:
        wp = prec + 20
    else:
        wp = prec + bitcount(gamma_size) + 20
    if mag < -wp:
        if type == 0:
            return mpf_sub(mpf_div(fone, x, wp), mpf_shift(fone, -wp), prec, rnd)
        if type == 1:
            return mpf_sub(fone, x, prec, rnd)
        if type == 2:
            return mpf_add(x, mpf_shift(fone, mag - wp), prec, rnd)
        if type == 3:
            return mpf_neg(mpf_log(mpf_abs(x), prec, rnd))
    if type == 1:
        return mpf_gamma(mpf_add(x, fone), prec, rnd, 0)
    if exp >= -1:
        if is_integer:
            if gamma_size < 10 * wp:
                if type == 0:
                    return from_int(ifac(n - 1), prec, rnd)
                if type == 2:
                    return from_rational(MPZ_ONE, ifac(n - 1), prec, rnd)
                if type == 3:
                    return mpf_log(from_int(ifac(n - 1)), prec, rnd)
        if n < 100 or gamma_size < 10 * wp:
            if sign:
                w = sqrtpi_fixed(wp)
                if n % 2:
                    f = ifac2(2 * n + 1)
                else:
                    f = -ifac2(2 * n + 1)
                if type == 0:
                    return mpf_shift(from_rational(w, f, prec, rnd), -wp + n + 1)
                if type == 2:
                    return mpf_shift(from_rational(f, w, prec, rnd), wp - n - 1)
                if type == 3:
                    return mpf_log(mpf_shift(from_rational(w, abs(f), prec, rnd), -wp + n + 1), prec, rnd)
            elif n == 0:
                if type == 0:
                    return mpf_sqrtpi(prec, rnd)
                if type == 2:
                    return mpf_div(fone, mpf_sqrtpi(wp), prec, rnd)
                if type == 3:
                    return mpf_log(mpf_sqrtpi(wp), prec, rnd)
            else:
                w = sqrtpi_fixed(wp)
                w = from_man_exp(w * ifac2(2 * n - 1), -wp - n)
                if type == 0:
                    return mpf_pos(w, prec, rnd)
                if type == 2:
                    return mpf_div(fone, w, prec, rnd)
                if type == 3:
                    return mpf_log(mpf_abs(w), prec, rnd)
    offset = exp + wp
    if offset >= 0:
        absxman = man << offset
    else:
        absxman = man >> -offset
    if type == 3 and (not sign):
        one = MPZ_ONE << wp
        one_dist = abs(absxman - one)
        two_dist = abs(absxman - 2 * one)
        cancellation = wp - bitcount(min(one_dist, two_dist))
        if cancellation > 10:
            xsub1 = mpf_sub(fone, x)
            xsub2 = mpf_sub(ftwo, x)
            xsub1mag = xsub1[2] + xsub1[3]
            xsub2mag = xsub2[2] + xsub2[3]
            if xsub1mag < -wp:
                return mpf_mul(mpf_euler(wp), mpf_sub(fone, x), prec, rnd)
            if xsub2mag < -wp:
                return mpf_mul(mpf_sub(fone, mpf_euler(wp)), mpf_sub(x, ftwo), prec, rnd)
            wp += max(-xsub1mag, -xsub2mag)
            offset = exp + wp
            if offset >= 0:
                absxman = man << offset
            else:
                absxman = man >> -offset
    n_for_stirling = int(GAMMA_STIRLING_BETA * wp)
    if n < max(100, n_for_stirling) and wp < MAX_GAMMA_TAYLOR_PREC:
        if sign:
            absxman = -absxman
        return gamma_fixed_taylor(x, absxman, wp, prec, rnd, type)
    xorig = x
    r = 0
    if n < n_for_stirling:
        r = one = MPZ_ONE << wp
        d = n_for_stirling - n
        for k in xrange(d):
            r = r * absxman >> wp
            absxman += one
        x = xabs = from_man_exp(absxman, -wp)
        if sign:
            x = mpf_neg(x)
    else:
        xabs = mpf_abs(x)
    y = real_stirling_series(absxman, wp)
    u = to_fixed(mpf_log(xabs, wp), wp)
    u = (absxman - (MPZ_ONE << wp - 1)) * u >> wp
    y += u
    w = from_man_exp(y, -wp)
    if sign:
        A = mpf_mul(mpf_sin_pi(xorig, wp), xorig, wp)
        B = mpf_neg(mpf_pi(wp))
        if type == 0 or type == 2:
            A = mpf_mul(A, mpf_exp(w, wp))
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            if type == 0:
                return mpf_div(B, A, prec, rnd)
            if type == 2:
                return mpf_div(A, B, prec, rnd)
        if type == 3:
            if r:
                B = mpf_mul(B, from_man_exp(r, -wp), wp)
            A = mpf_add(mpf_log(mpf_abs(A), wp), w, wp)
            return mpf_sub(mpf_log(mpf_abs(B), wp), A, prec, rnd)
    else:
        if type == 0:
            if r:
                return mpf_div(mpf_exp(w, wp), from_man_exp(r, -wp), prec, rnd)
            return mpf_exp(w, prec, rnd)
        if type == 2:
            if r:
                return mpf_div(from_man_exp(r, -wp), mpf_exp(w, wp), prec, rnd)
            return mpf_exp(mpf_neg(w), prec, rnd)
        if type == 3:
            if r:
                return mpf_sub(w, mpf_log(from_man_exp(r, -wp), wp), prec, rnd)
            return mpf_pos(w, prec, rnd)