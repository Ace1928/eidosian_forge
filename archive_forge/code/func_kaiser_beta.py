import math
from cupy.fft import fft, ifft
from cupy.linalg import solve, lstsq, LinAlgError
from cupyx.scipy.linalg import toeplitz, hankel
import cupyx
from cupyx.scipy.signal.windows import get_window
import cupy
import numpy
def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.

    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.

    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.

    See Also
    --------
    scipy.signal.kaiser_beta

    """
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta