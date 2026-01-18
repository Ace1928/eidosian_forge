import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def ellipap(N, rp, rs):
    """Return (z,p,k) of Nth-order elliptic analog lowpass filter.

    The filter is a normalized prototype that has `rp` decibels of ripple
    in the passband and a stopband `rs` decibels down.

    The filter's angular (e.g., rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    ellip : Filter design function using this prototype
    scipy.signal.elliap

    """
    if abs(int(N)) != N:
        raise ValueError('Filter order must be a nonnegative integer')
    elif N == 0:
        return (cupy.array([]), cupy.array([]), 10 ** (-rp / 20))
    elif N == 1:
        p = -cupy.sqrt(1.0 / _pow10m1(0.1 * rp))
        k = -p
        z = []
        return (cupy.asarray(z), cupy.asarray(p), k)
    eps_sq = _pow10m1(0.1 * rp)
    eps = cupy.sqrt(eps_sq)
    ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    if ck1_sq == 0:
        raise ValueError('Cannot design a filter with given rp and rs specifications.')
    m = _ellipdeg(N, ck1_sq)
    capk = special.ellipk(m)
    j = cupy.arange(1 - N % 2, N, 2)
    EPSILON = 2e-16
    s, c, d, phi = special.ellipj(j * capk / N, m * cupy.ones_like(j))
    snew = cupy.compress(cupy.abs(s) > EPSILON, s, axis=-1)
    z = 1j / (cupy.sqrt(m) * snew)
    z = cupy.concatenate((z, z.conj()))
    r = _arc_jac_sc1(1.0 / eps, ck1_sq)
    v0 = capk * r / (N * special.ellipk(ck1_sq))
    sv, cv, dv, phi = special.ellipj(v0, 1 - m)
    p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2.0)
    if N % 2:
        mask = cupy.abs(p.imag) > EPSILON * cupy.sqrt((p * p.conj()).sum(axis=0).real)
        newp = cupy.compress(mask, p, axis=-1)
        p = cupy.concatenate((p, newp.conj()))
    else:
        p = cupy.concatenate((p, p.conj()))
    k = (cupy.prod(-p, axis=0) / cupy.prod(-z, axis=0)).real
    if N % 2 == 0:
        k = k / cupy.sqrt(1 + eps_sq)
    return (z, p, k)