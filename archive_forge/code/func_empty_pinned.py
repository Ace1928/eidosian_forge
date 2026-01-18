import numpy
from cupy import cuda
from cupy._creation.basic import _new_like_order_and_strides
from cupy._core import internal
def empty_pinned(shape, dtype=float, order='C'):
    """Returns a new, uninitialized NumPy array with the given shape
    and dtype.

    This is a convenience function which is just :func:`numpy.empty`,
    except that the underlying memory is pinned/pagelocked.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Returns:
        numpy.ndarray: A new array with elements not initialized.

    .. seealso:: :func:`numpy.empty`

    """
    shape = _update_shape(None, shape)
    nbytes = internal.prod(shape) * numpy.dtype(dtype).itemsize
    mem = cuda.alloc_pinned_memory(nbytes)
    out = numpy.ndarray(shape, dtype=dtype, buffer=mem, order=order)
    return out