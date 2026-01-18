from CuSignal under terms of the MIT license.
import warnings
from typing import Set
import cupy
import numpy as np
def boxcar(M, sym=True):
    """Return a boxcar or rectangular window.

    Also known as a rectangular window or Dirichlet window, this is equivalent
    to no window at all.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        Whether the window is symmetric. (Has no effect for boxcar.)

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1.

    Examples
    --------
    Plot the window and its frequency response:

    >>> from cupyx.scipy.signal.windows import boxcar
    >>> import cupy
    >>> from cupy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = boxcar(51)
    >>> plt.plot(cupy.asnumpy(window))
    >>> plt.title("Boxcar window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = cupy.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * cupy.log10(cupy.abs(fftshift(A / cupy.abs(A).max())))
    >>> plt.plot(cupy.asnumpy(freq), cupy.asnumpy(response))
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the boxcar window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if _len_guards(M):
        return cupy.ones(M)
    M, needs_trunc = _extend(M, sym)
    w = cupy.ones(M, dtype=cupy.float64)
    return _truncate(w, needs_trunc)