import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
def mpc_psi(m, z, prec, rnd=round_fast):
    """
    Computation of the polygamma function of arbitrary integer order
    m >= 0, for a complex argument z.
    """
    if m == 0:
        return mpc_psi0(z, prec, rnd)
    re, im = z
    wp = prec + 20
    sign, man, exp, bc = re
    if not im[1]:
        if im in (finf, fninf, fnan):
            return (fnan, fnan)
    if not man:
        if re == finf and im == fzero:
            return (fzero, fzero)
        if re == fnan:
            return (fnan, fnan)
    w = to_int(re)
    n = int(0.4 * wp + 4 * m)
    s = mpc_zero
    if w < n:
        for k in xrange(w, n):
            t = mpc_pow_int(z, -m - 1, wp)
            s = mpc_add(s, t, wp)
            z = mpc_add_mpf(z, fone, wp)
    zm = mpc_pow_int(z, -m, wp)
    z2 = mpc_pow_int(z, -2, wp)
    integral_term = mpc_div_mpf(zm, from_int(m), wp)
    s = mpc_add(s, integral_term, wp)
    s = mpc_add(s, mpc_mul_mpf(mpc_div(zm, z, wp), fhalf, wp), wp)
    a = m + 1
    b = 2
    k = 1
    magn = mpc_abs(s, 10)
    magn = magn[2] + magn[3]
    eps = mpf_shift(fone, magn - wp + 2)
    while 1:
        zm = mpc_mul(zm, z2, wp)
        bern = mpf_bernoulli(2 * k, wp)
        scal = mpf_mul_int(bern, a, wp)
        scal = mpf_div(scal, from_int(b), wp)
        term = mpc_mul_mpf(zm, scal, wp)
        s = mpc_add(s, term, wp)
        szterm = mpc_abs(term, 10)
        if k > 2 and mpf_le(szterm, eps):
            break
        a *= (m + 2 * k) * (m + 2 * k + 1)
        b *= (2 * k + 1) * (2 * k + 2)
        k += 1
    v = mpc_mul_mpf(s, mpf_gamma(from_int(m + 1), wp), prec, rnd)
    if not m & 1:
        v = (mpf_neg(v[0]), mpf_neg(v[1]))
    return v