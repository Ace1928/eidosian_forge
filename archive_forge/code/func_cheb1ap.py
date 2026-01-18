import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def cheb1ap(N, rp):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rp` decibels of ripple in the passband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    cheby1 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError('Filter order must be a nonnegative integer')
    elif N == 0:
        return (cupy.array([]), cupy.array([]), 10 ** (-rp / 20))
    z = cupy.array([])
    eps = cupy.sqrt(10 ** (0.1 * rp) - 1.0)
    mu = 1.0 / N * cupy.arcsinh(1 / eps)
    m = cupy.arange(-N + 1, N, 2)
    theta = pi * m / (2 * N)
    p = -cupy.sinh(mu + 1j * theta)
    k = cupy.prod(-p, axis=0).real
    if N % 2 == 0:
        k = k / cupy.sqrt(1 + eps * eps)
    return (z, p, k)