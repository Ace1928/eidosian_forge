import cupy
from cupy import _core
from cupy._core import fusion
from cupy import _util
from cupy._core import _routines_indexing as _indexing
from cupy._core import _routines_statistics as _statistics
def argwhere(a):
    """Return the indices of the elements that are non-zero.

    Returns a (N, ndim) dimantional array containing the
    indices of the non-zero elements. Where `N` is number of
    non-zero elements and `ndim` is dimension of the given array.

    Args:
        a (cupy.ndarray): array

    Returns:
        cupy.ndarray: Indices of elements that are non-zero.

    .. seealso:: :func:`numpy.argwhere`

    """
    _util.check_array(a, arg_name='a')
    return _indexing._ndarray_argwhere(a)