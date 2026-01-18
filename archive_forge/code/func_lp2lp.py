import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def lp2lp(b, a, wo=1.0):
    """
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g. rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed low-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed low-pass filter.

    See Also
    --------
    lp2hp, lp2bp, lp2bs, bilinear
    lp2lp_zpk
    scipy.signal.lp2lp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \\rightarrow \\frac{s}{\\omega_0}

    """
    a, b = map(cupy.atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    M = max(d, n)
    pwo = wo ** cupy.arange(M - 1, -1, -1)
    start1 = max((n - d, 0))
    start2 = max((d - n, 0))
    b = b * pwo[start1] / pwo[start2:]
    a = a * pwo[start1] / pwo[start1:]
    return normalize(b, a)