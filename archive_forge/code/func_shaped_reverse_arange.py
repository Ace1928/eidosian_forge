import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings
import numpy
import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def shaped_reverse_arange(shape, xp=cupy, dtype=numpy.float32):
    """Returns an array filled with decreasing numbers.

    Args:
         shape(tuple of int): Shape of returned ndarray.
         xp(numpy or cupy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.

    Returns:
         numpy.ndarray or cupy.ndarray:
         The array filled with :math:`N, \\cdots, 1` with specified dtype
         with given shape, array module.
         Here, :math:`N` is the size of the returned array.
         If ``dtype`` is ``numpy.bool_``, evens (resp. odds) are converted to
         ``True`` (resp. ``False``).
    """
    dtype = numpy.dtype(dtype)
    size = internal.prod(shape)
    a = numpy.arange(size, 0, -1)
    if dtype == '?':
        a = a % 2 == 0
    elif dtype.kind == 'c':
        a = a + a * 1j
    return xp.array(a.astype(dtype).reshape(shape))