from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def general_gaussian(M, p, sig, sym=True):
    """Return a window with a generalized Gaussian shape.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    p : float
        Shape parameter.  p = 1 is identical to `gaussian`, p = 0.5 is
        the same shape as the Laplace distribution.
    sig : float
        The standard deviation, sigma.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Notes
    -----
    The generalized Gaussian window is defined as

    .. math::  w(n) = e^{ -\\frac{1}{2}\\left|\\frac{n}{\\sigma}\\right|^{2p} }

    the half-power point is at

    .. math::  (2 \\log(2))^{1/(2 p)} \\sigma

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cupyx.scipy.signal.windows
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cupyx.scipy.signal.windows.general_gaussian(51, p=1.5, sig=7)
    >>> plt.plot(cupy.asnumpy(window))
    >>> plt.title(r"Generalized Gaussian window (p=1.5, $\\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cupy.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cupy.log10(cupy.abs(fftshift(A / cupy.abs(A).max())))
    >>> plt.plot(cupy.asnumpy(freq), cupy.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Freq. resp. of the gen. Gaussian "
    ...           r"window (p=1.5, $\\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cupy.ones(M)
    M, needs_trunc = _extend(M, sym)
    w = _general_gaussian_kernel(p, sig, size=M)
    return _truncate(w, needs_trunc)