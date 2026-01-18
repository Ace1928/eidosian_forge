import numpy
from cupy import _core
from cupy._core import fusion
def frombuffer(*args, **kwargs):
    """Interpret a buffer as a 1-dimensional array.

    .. note::
        Uses NumPy's ``frombuffer`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.frombuffer`

    """
    return asarray(numpy.frombuffer(*args, **kwargs))