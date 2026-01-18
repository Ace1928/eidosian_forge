import numpy
from cupy import _core
from cupy._core import fusion
def genfromtxt(*args, **kwargs):
    """Load data from text file, with missing values handled as specified.

    .. note::
        Uses NumPy's ``genfromtxt`` and coerces the result to a CuPy array.

    .. seealso:: :func:`numpy.genfromtxt`
    """
    return asarray(numpy.genfromtxt(*args, **kwargs))