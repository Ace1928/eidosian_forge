from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve
from scipy.fft._pocketfft.helper import _datacopied
def itilbert(x, h, period=None, _cache=_cache):
    """
    Return inverse h-Tilbert transform of a periodic sequence x.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j
      y_0 = 0

    For more details, see `tilbert`.

    """
    tmp = asarray(x)
    if iscomplexobj(tmp):
        return itilbert(tmp.real, h, period) + 1j * itilbert(tmp.imag, h, period)
    if period is not None:
        h = h * 2 * pi / period
    n = len(x)
    omega = _cache.get((n, h))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, h=h):
            if k:
                return -tanh(h * k)
            return 0
        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[n, h] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp, omega, swap_real_imag=1, overwrite_x=overwrite_x)