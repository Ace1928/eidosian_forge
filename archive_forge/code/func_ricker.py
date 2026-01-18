import cupy
import numpy as np
from cupyx.scipy.signal._signaltools import convolve
def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:

        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,

    where ``A = 2/sqrt(3a)pi^1/4``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Examples
    --------
    >>> import cupyx.scipy.signal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    >>> points = 100
    >>> a = 4.0
    >>> vec2 = cupyx.scipy.signal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(cupy.asnumpy(vec2))
    >>> plt.show()

    """
    return _ricker_kernel(a, size=int(points))