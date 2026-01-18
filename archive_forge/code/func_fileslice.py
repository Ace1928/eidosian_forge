import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def fileslice(fileobj, sliceobj, shape, dtype, offset=0, order='C', heuristic=threshold_heuristic, lock=None):
    """Slice array in `fileobj` using `sliceobj` slicer and array definitions

    `fileobj` contains the contiguous binary data for an array ``A`` of shape,
    dtype, memory layout `shape`, `dtype`, `order`, with the binary data
    starting at file offset `offset`.

    Our job is to return the sliced array ``A[sliceobj]`` in the most efficient
    way in terms of memory and time.

    Sometimes it will be quicker to read memory that we will later throw away,
    to save time we might lose doing short seeks on `fileobj`.  Call these
    alternatives: (read + discard); and skip.  This routine guesses when to
    (read+discard) or skip using the callable `heuristic`, with a default using
    a hard threshold for the memory gap large enough to prefer a skip.

    Parameters
    ----------
    fileobj : file-like object
        file-like object, opened for reading in binary mode. Implements
        ``read`` and ``seek``.
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``.
    shape : sequence
        shape of full array inside `fileobj`.
    dtype : dtype specifier
        dtype of array inside `fileobj`, or input to ``numpy.dtype`` to specify
        array dtype.
    offset : int, optional
        offset of array data within `fileobj`
    order : {'C', 'F'}, optional
        memory layout of array in `fileobj`.
    heuristic : callable, optional
        function taking slice object, axis length, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer` and see :func:`threshold_heuristic` for an
        example.
    lock : {None, threading.Lock, lock-like} optional
        If provided, used to ensure that paired calls to ``seek`` and ``read``
        cannot be interrupted by another thread accessing the same ``fileobj``.
        Each thread which accesses the same file via ``read_segments`` must
        share a lock in order to ensure that the file access is thread-safe.
        A lock does not need to be provided for single-threaded access. The
        default value (``None``) results in a lock-like object  (a
        ``_NullLock``) which does not do anything.

    Returns
    -------
    sliced_arr : array
        Array in `fileobj` as sliced with `sliceobj`
    """
    if is_fancy(sliceobj):
        raise ValueError('Cannot handle fancy indexing')
    dtype = np.dtype(dtype)
    itemsize = int(dtype.itemsize)
    segments, sliced_shape, post_slicers = calc_slicedefs(sliceobj, shape, itemsize, offset, order)
    n_bytes = reduce(operator.mul, sliced_shape, 1) * itemsize
    arr_data = read_segments(fileobj, segments, n_bytes, lock)
    sliced = np.ndarray(sliced_shape, dtype, buffer=arr_data, order=order)
    return sliced[post_slicers]