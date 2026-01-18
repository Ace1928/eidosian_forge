from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def chebwin(M, at, sym=True):
    """Return a Dolph-Chebyshev window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    at : float
        Attenuation (in dB).
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value always normalized to 1

    Notes
    -----
    This window optimizes for the narrowest main lobe width for a given order
    `M` and sidelobe equiripple attenuation `at`, using Chebyshev
    polynomials.  It was originally developed by Dolph to optimize the
    directionality of radio antenna arrays.

    Unlike most windows, the Dolph-Chebyshev is defined in terms of its
    frequency response:

    .. math:: W(k) = \\frac
              {\\cos\\{M \\cos^{-1}[\\beta \\cos(\\frac{\\pi k}{M})]\\}}
              {\\cosh[M \\cosh^{-1}(\\beta)]}

    where

    .. math:: \\beta = \\cosh \\left [\\frac{1}{M}
              \\cosh^{-1}(10^\\frac{A}{20}) \\right ]

    and 0 <= abs(k) <= M-1. A is the attenuation in decibels (`at`).

    The time domain window is then generated using the IFFT, so
    power-of-two `M` are the fastest to generate, and prime number `M` are
    the slowest.

    The equiripple condition in the frequency domain creates impulses in the
    time domain, which appear at the ends of the window.

    For more information, see [1]_, [2]_ and [3]_

    References
    ----------
    .. [1] C. Dolph, "A current distribution for broadside arrays which
           optimizes the relationship between beam width and side-lobe level",
           Proceedings of the IEEE, Vol. 34, Issue 6
    .. [2] Peter Lynch, "The Dolph-Chebyshev Window: A Simple Optimal Filter",
           American Meteorological Society (April 1997)
           http://mathsci.ucd.ie/~plynch/Publications/Dolph.pdf
    .. [3] F. J. Harris, "On the use of windows for harmonic analysis with the
           discrete Fourier transforms", Proceedings of the IEEE, Vol. 66,
           No. 1, January 1978

    Examples
    --------
    Plot the window and its frequency response:

    >>> import cupyx.scipy.signal.windows
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = cupyx.scipy.signal.windows.chebwin(51, at=100)
    >>> plt.plot(cupy.asnumpy(window))
    >>> plt.title("Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cupy.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cupy.log10(cupy.abs(fftshift(A / cupy.abs(A).max())))
    >>> plt.plot(cupy.asnumpy(freq), cupy.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Dolph-Chebyshev window (100 dB)")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if abs(at) < 45:
        warnings.warn('This window is not suitable for spectral analysis for attenuation values lower than about 45dB because the equivalent noise bandwidth of a Chebyshev window does not grow monotonically with increasing sidelobe attenuation when the attenuation is smaller than about 45 dB.')
    if _len_guards(M):
        return cupy.ones(M)
    M, needs_trunc = _extend(M, sym)
    order = M - 1.0
    beta = np.cosh(1.0 / order * np.arccosh(10 ** (abs(at) / 20.0)))
    p = _chebwin_kernel(order, beta, size=M)
    if M % 2:
        w = cupy.real(cupy.fft.fft(p))
        n = (M + 1) // 2
        w = w[:n]
        w = cupy.concatenate((w[n - 1:0:-1], w))
    else:
        w = cupy.real(cupy.fft.fft(p))
        n = M // 2 + 1
        w = cupy.concatenate((w[n - 1:0:-1], w[1:n]))
    w = w / cupy.max(w)
    return _truncate(w, needs_trunc)