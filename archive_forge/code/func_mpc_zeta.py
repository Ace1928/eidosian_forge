import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpc_zeta(s, prec, rnd=round_fast, alt=0, force=False):
    re, im = s
    if im == fzero:
        return (mpf_zeta(re, prec, rnd, alt), fzero)
    if not force and mpf_gt(mpc_abs(s, 10), from_int(prec)):
        raise NotImplementedError
    wp = prec + 20
    r = mpc_sub(mpc_one, s, wp)
    asign, aman, aexp, abc = mpc_abs(r, 10)
    pole_dist = -2 * (aexp + abc)
    if pole_dist > wp:
        if alt:
            q = mpf_ln2(wp)
            y = mpf_mul(q, mpf_euler(wp), wp)
            g = mpf_shift(mpf_mul(q, q, wp), -1)
            g = mpf_sub(y, g)
            z = mpc_mul_mpf(r, mpf_neg(g), wp)
            z = mpc_add_mpf(z, q, wp)
            return mpc_pos(z, prec, rnd)
        else:
            q = mpc_neg(mpc_div(mpc_one, r, wp))
            q = mpc_add_mpf(q, mpf_euler(wp), wp)
            return mpc_pos(q, prec, rnd)
    else:
        wp += max(0, pole_dist)
    if mpf_lt(re, fzero):
        if alt:
            q = mpc_sub(mpc_one, mpc_pow(mpc_two, mpc_sub(mpc_one, s, wp), wp), wp)
            return mpc_mul(mpc_zeta(s, wp), q, prec, rnd)
        y = mpc_sub(mpc_one, s, 10 * wp)
        a = mpc_gamma(y, wp)
        b = mpc_zeta(y, wp)
        c = mpc_sin_pi(mpc_shift(s, -1), wp)
        rsign, rman, rexp, rbc = re
        isign, iman, iexp, ibc = im
        mag = max(rexp + rbc, iexp + ibc)
        wp2 = wp + max(0, mag)
        pi = mpf_pi(wp + wp2)
        pi2 = (mpf_shift(pi, 1), fzero)
        d = mpc_div_mpf(mpc_pow(pi2, s, wp2), pi, wp2)
        return mpc_mul(a, mpc_mul(b, mpc_mul(c, d, wp), wp), prec, rnd)
    n = int(wp / 2.54 + 5)
    n += int(0.9 * abs(to_int(im)))
    d = borwein_coefficients(n)
    ref = to_fixed(re, wp)
    imf = to_fixed(im, wp)
    tre = MPZ_ZERO
    tim = MPZ_ZERO
    one = MPZ_ONE << wp
    one_2wp = MPZ_ONE << 2 * wp
    critical_line = re == fhalf
    ln2 = ln2_fixed(wp)
    pi2 = pi_fixed(wp - 1)
    wp2 = wp + wp
    for k in xrange(n):
        log = log_int_fixed(k + 1, wp, ln2)
        if critical_line:
            w = one_2wp // isqrt_fast(k + 1 << wp2)
        else:
            w = exp_fixed(-ref * log >> wp, wp)
        if k & 1:
            w *= d[n] - d[k]
        else:
            w *= d[k] - d[n]
        wre, wim = cos_sin_fixed(-imf * log >> wp, wp, pi2)
        tre += w * wre >> wp
        tim += w * wim >> wp
    tre //= -d[n]
    tim //= -d[n]
    tre = from_man_exp(tre, -wp, wp)
    tim = from_man_exp(tim, -wp, wp)
    if alt:
        return mpc_pos((tre, tim), prec, rnd)
    else:
        q = mpc_sub(mpc_one, mpc_pow(mpc_two, r, wp), wp)
        return mpc_div((tre, tim), q, prec, rnd)