import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
@_wraps_polyroutine
def polyadd(a1, a2):
    """Computes the sum of two polynomials.

    Args:
        a1 (scalar, cupy.ndarray or cupy.poly1d): first input polynomial.
        a2 (scalar, cupy.ndarray or cupy.poly1d): second input polynomial.

    Returns:
        cupy.ndarray or cupy.poly1d: The sum of the inputs.

    .. seealso:: :func:`numpy.polyadd`

    """
    if a1.size < a2.size:
        a1, a2 = (a2, a1)
    out = cupy.pad(a2, (a1.size - a2.size, 0))
    out = out.astype(cupy.result_type(a1, a2), copy=False)
    out += a1
    return out