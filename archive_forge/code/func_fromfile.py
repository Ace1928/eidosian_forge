import numpy
from cupy import _core
from cupy._core import fusion
def fromfile(*args, **kwargs):
    """Reads an array from a file.

    .. note::
        Uses NumPy's ``fromfile`` and coerces the result to a CuPy array.

    .. note::
       If you let NumPy's ``fromfile`` read the file in big-endian, CuPy
       automatically swaps its byte order to little-endian, which is the NVIDIA
       and AMD GPU architecture's native use.

    .. seealso:: :func:`numpy.fromfile`

    """
    return asarray(numpy.fromfile(*args, **kwargs))