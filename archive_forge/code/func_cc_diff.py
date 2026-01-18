from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve
from scipy.fft._pocketfft.helper import _datacopied
def cc_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-cosh/cosh pseudo-derivative of a periodic sequence.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = cosh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a,b : float
        Defines the parameters of the sinh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is ``2*pi``.

    Returns
    -------
    cc_diff : ndarray
        Pseudo-derivative of periodic sequence `x`.

    Notes
    -----
    ``cc_diff(cc_diff(x,a,b),b,a) == x``

    """
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return cc_diff(tmp.real, a, b, period) + 1j * cc_diff(tmp.imag, a, b, period)
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
            return cosh(a * k) / cosh(b * k)
        omega = convolve.init_convolution_kernel(n, kernel)
        _cache[n, a, b] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, overwrite_x=overwrite_x)