from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve
from scipy.fft._pocketfft.helper import _datacopied
def ihilbert(x):
    """
    Return inverse Hilbert transform of a periodic sequence x.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*sign(j) * x_j
      y_0 = 0

    """
    return -hilbert(x)