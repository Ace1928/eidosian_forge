import numpy
from cupy import cuda
from cupy._creation.basic import _new_like_order_and_strides
from cupy._core import internal
def empty_like_pinned(a, dtype=None, order='K', subok=None, shape=None):
    """Returns a new, uninitialized NumPy array with the same shape and dtype
    as those of the given array.

    This is a convenience function which is just :func:`numpy.empty_like`,
    except that the underlying memory is pinned/pagelocked.

    This function currently does not support ``subok`` option.

    Args:
        a (numpy.ndarray or cupy.ndarray): Base array.
        dtype: Data type specifier. The data type of ``a`` is used by default.
        order ({'C', 'F', 'A', or 'K'}): Overrides the memory layout of the
            result. ``'C'`` means C-order, ``'F'`` means F-order, ``'A'`` means
            ``'F'`` if ``a`` is Fortran contiguous, ``'C'`` otherwise.
            ``'K'`` means match the layout of ``a`` as closely as possible.
        subok: Not supported yet, must be None.
        shape (int or tuple of ints): Overrides the shape of the result. If
            ``order='K'`` and the number of dimensions is unchanged, will try
            to keep order, otherwise, ``order='C'`` is implied.

    Returns:
        numpy.ndarray: A new array with same shape and dtype of ``a`` with
        elements not initialized.

    .. seealso:: :func:`numpy.empty_like`

    """
    if subok is not None:
        raise TypeError('subok is not supported yet')
    if dtype is None:
        dtype = a.dtype
    shape = _update_shape(a, shape)
    order, strides, _ = _new_like_order_and_strides(a, dtype, order, shape, get_memptr=False)
    nbytes = internal.prod(shape) * numpy.dtype(dtype).itemsize
    mem = cuda.alloc_pinned_memory(nbytes)
    out = numpy.ndarray(shape, dtype=dtype, buffer=mem, strides=strides, order=order)
    return out