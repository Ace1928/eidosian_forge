from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def gaussian(M, std, sym=True):
    """Return a Gaussian window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    std : float
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
    The Gaussian window is defined as

    .. math::  w(n) = e^{ -\\frac{1}{2}\\left(\\frac{n}{\\sigma}\\right)^2 }

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cupyx.scipy.signal.windows
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cupyx.scipy.signal.windows.gaussian(51, std=7)
    >>> plt.plot(cupy.asnumpy(window))
    >>> plt.title(r"Gaussian window ($\\sigma$=7)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cupy.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cupy.log10(cupy.abs(fftshift(A / cupy.abs(A).max())))
    >>> plt.plot(cupy.asnumpy(freq), cupy.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title(r"Frequency response of the Gaussian window ($\\sigma$=7)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cupy.ones(M)
    M, needs_trunc = _extend(M, sym)
    w = _gaussian_kernel(std, size=M)
    return _truncate(w, needs_trunc)