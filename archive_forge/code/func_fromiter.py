import numpy
from cupy import _core
from cupy._core import fusion
def fromiter(*args, **kwargs):
    """Create a new 1-dimensional array from an iterable object.

    .. note::
        Uses NumPy's ``fromiter`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.fromiter`
    """
    return asarray(numpy.fromiter(*args, **kwargs))