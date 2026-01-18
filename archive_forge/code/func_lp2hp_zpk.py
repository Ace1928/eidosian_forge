import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def lp2hp_zpk(z, p, k, wo=1.0):
    """
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2hp
    scipy.signal.lp2hp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \\rightarrow \\frac{\\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    """
    z = cupy.atleast_1d(z)
    p = cupy.atleast_1d(p)
    wo = float(wo)
    degree = _relative_degree(z, p)
    z_hp = wo / z
    p_hp = wo / p
    z_hp = cupy.append(z_hp, cupy.zeros(degree))
    k_hp = k * cupy.real(cupy.prod(-z) / cupy.prod(-p))
    return (z_hp, p_hp, k_hp)