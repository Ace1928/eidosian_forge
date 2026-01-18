from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def flattop(M, sym=True):
    """Return a flat top window.

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

    Notes
    -----
    Flat top windows are used for taking accurate measurements of signal
    amplitude in the frequency domain, with minimal scalloping error from the
    center of a frequency bin to its edges, compared to others.  This is a
    5th-order cosine window, with the 5 terms optimized to make the main lobe
    maximally flat. [1]_

    References
    ----------
    .. [1] D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for
           Measurement Systems", Springer Media, 2006, p. 70
           `10.1007/0-387-28666-7 <https://doi.org/10.1007/0-387-28666-7>`_

    Examples
    --------
    Plot the window and its frequency response:

    >>> from cupyx.scipy.signal.windows import flattop
    >>> import cupy as cp
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = flattop(51)
    >>> plt.plot(cupy.asnumpy(window))
    >>> plt.title("Flat top window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cupy.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cupy.log10(cupy.abs(fftshift(A / cupy.abs(A).max())))
    >>> plt.plot(cupy.asnumpy(freq), cupy.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the flat top window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    return general_cosine(M, a, sym)