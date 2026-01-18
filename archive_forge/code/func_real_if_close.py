import cupy
import cupyx.scipy.fft
from cupy import _core
from cupy._core import _routines_math as _math
from cupy._core import fusion
from cupy.lib import stride_tricks
import numpy
def real_if_close(a, tol=100):
    """If input is complex with all imaginary parts close to zero, return real
    parts.
    "Close to zero" is defined as `tol` * (machine epsilon of the type for
    `a`).

    .. warning::

            This function may synchronize the device.

    .. seealso:: :func:`numpy.real_if_close`
    """
    if not issubclass(a.dtype.type, cupy.complexfloating):
        return a
    if tol > 1:
        f = numpy.finfo(a.dtype.type)
        tol = f.eps * tol
    if cupy.all(cupy.absolute(a.imag) < tol):
        a = a.real
    return a