import numpy
from cupy import _core
from cupy._core import fusion
def loadtxt(*args, **kwargs):
    """Load data from a text file.

    .. note::
        Uses NumPy's ``loadtxt`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.loadtxt`
    """
    return asarray(numpy.loadtxt(*args, **kwargs))