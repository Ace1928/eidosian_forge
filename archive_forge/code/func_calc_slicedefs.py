import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def calc_slicedefs(sliceobj, in_shape, itemsize, offset, order, heuristic=threshold_heuristic):
    """Return parameters for slicing array with `sliceobj` given memory layout

    Calculate the best combination of skips / (read + discard) to use for
    reading the data from disk / memory, then generate corresponding
    `segments`, the disk offsets and read lengths to read the memory.  If we
    have chosen some (read + discard) optimization, then we need to discard the
    surplus values from the read array using `post_slicers`, a slicing tuple
    that takes the array as read from a file-like object, and returns the array
    we want.

    Parameters
    ----------
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    in_shape : sequence
        shape of underlying array to be sliced
    itemsize : int
        element size in array (in bytes)
    offset : int
        offset of array data in underlying file or memory buffer
    order : {'C', 'F'}
        memory layout of underlying array
    heuristic : callable, optional
        function taking slice object, dim_len, stride length as arguments,
        returning one of 'full', 'contiguous', None.  See
        :func:`optimize_slicer` and :func:`threshold_heuristic`

    Returns
    -------
    segments : list
        list of 2 element lists where lists are (offset, length), giving
        absolute memory offset in bytes and number of bytes to read
    read_shape : tuple
        shape with which to interpret memory as read from `segments`.
        Interpreting the memory read from `segments` with this shape, and a
        dtype, gives an intermediate array - call this ``R``
    post_slicers : tuple
        Any new slicing to be applied to the array ``R`` after reading via
        `segments` and reshaping via `read_shape`.  Slices are in terms of
        `read_shape`.  If empty, no new slicing to apply
    """
    if order not in 'CF':
        raise ValueError("order should be one of 'CF'")
    sliceobj = canonical_slicers(sliceobj, in_shape)
    if order == 'C':
        sliceobj = sliceobj[::-1]
        in_shape = in_shape[::-1]
    read_slicers, post_slicers = optimize_read_slicers(sliceobj, in_shape, itemsize, heuristic)
    segments = slicers2segments(read_slicers, in_shape, offset, itemsize)
    if all((s == slice(None) for s in post_slicers)):
        post_slicers = []
    read_shape = predict_shape(read_slicers, in_shape)
    if order == 'C':
        read_shape = read_shape[::-1]
        post_slicers = post_slicers[::-1]
    return (list(segments), tuple(read_shape), tuple(post_slicers))