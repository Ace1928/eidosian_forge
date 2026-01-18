import cupy
import numpy as np
from cupyx.scipy.signal._signaltools import convolve
def morlet(M, w=5.0, s=1.0, complete=True):
    """
    Complex Morlet wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0. Default is 5
    s : float, optional
        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
    complete : bool, optional
        Whether to use the complete or the standard version.

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    cupyx.scipy.signal.gausspulse

    Notes
    -----
    The standard version::

        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

    This commonly used wavelet is often referred to simply as the
    Morlet wavelet.  Note that this simplified version can cause
    admissibility problems at low values of `w`.

    The complete version::

        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

    This version has a correction
    term to improve admissibility. For `w` greater than 5, the
    correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to `s`.

    The fundamental frequency of this wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.

    Note: This function was created before `cwt` and is not compatible
    with it.

    """
    return _morlet_kernel(w, s, complete, size=M)