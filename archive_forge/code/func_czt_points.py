import cmath
import numbers
import cupy
from numpy import pi
from cupyx.scipy.fft import fft, ifft, next_fast_len
def czt_points(m, w=None, a=1 + 0j):
    """
    Return the points at which the chirp z-transform is computed.

    Parameters
    ----------
    m : int
        The number of points desired.
    w : complex, optional
        The ratio between points in each step.
        Defaults to equally spaced points around the entire unit circle.
    a : complex, optional
        The starting point in the complex plane.  Default is 1+0j.

    Returns
    -------
    out : ndarray
        The points in the Z plane at which `CZT` samples the z-transform,
        when called with arguments `m`, `w`, and `a`, as complex numbers.

    See Also
    --------
    CZT : Class that creates a callable chirp z-transform function.
    czt : Convenience function for quickly calculating CZT.
    scipy.signal.czt_points

    """
    m = _validate_sizes(1, m)
    k = cupy.arange(m)
    a = 1.0 * a
    if w is None:
        return a * cupy.exp(2j * pi * k / m)
    else:
        w = 1.0 * w
        return a * w ** (-k)