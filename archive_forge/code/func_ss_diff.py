from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve
from scipy.fft._pocketfft.helper import _datacopied
def ss_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = sinh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
      y_0 = a/b * x_0

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a,b
        Defines the parameters of the sinh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is ``2*pi``.

    Notes
    -----
    ``ss_diff(ss_diff(x,a,b),b,a) == x``

    """
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return ss_diff(tmp.real, a, b, period) + 1j * ss_diff(tmp.imag, a, b, period)
    if period is not None:
        a = a * 2 * pi / period
        b = b * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, a, b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, a=a, b=b):
            if k:
                return sinh(a * k) / sinh(b * k)
            return float(a) / b
        omega = convolve.init_convolution_kernel(n, kernel)
        _cache[n, a, b] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, overwrite_x=overwrite_x)