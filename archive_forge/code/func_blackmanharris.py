from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def blackmanharris(M, sym=True):
    """Return a minimum 4-term Blackman-Harris window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from cupyx.scipy.signal.windows import blackmanharris
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = blackmanharris(51)
    >>> plt.plot(cupy.asnumpy(window))
    >>> plt.title("Blackman-Harris window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cupy.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cupy.log10(cupy.abs(fftshift(A / cupy.abs(A).max())))
    >>> plt.plot(cupy.asnumpy(freq), cupy.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman-Harris window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    return general_cosine(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)