import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def lp2bs(b, a, wo=1.0, bw=1.0):
    """
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-stop filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-stop
        filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, bilinear
    lp2bs_zpk
    scipy.signal.lp2bs

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \\rightarrow \\frac{s \\cdot \\mathrm{BW}}{s^2 + {\\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.
    """
    a, b = map(cupy.atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = cupy.mintypecode((a.dtype, b.dtype))
    M = max(N, D)
    Np = M + M
    Dp = M + M
    bprime = cupy.empty(Np + 1, artype)
    aprime = cupy.empty(Dp + 1, artype)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += comb(M - i, k) * b[N - i] * wosq ** (M - i - k) * bw ** i
        bprime[Np - j] = val
    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += comb(M - i, k) * a[D - i] * wosq ** (M - i - k) * bw ** i
        aprime[Dp - j] = val
    return normalize(bprime, aprime)