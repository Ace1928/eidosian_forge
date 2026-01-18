import functools
import inspect
import os
import random
from typing import Tuple, Type
import traceback
import unittest
import warnings
import numpy
import cupy
from cupy.testing import _array
from cupy.testing import _parameterized
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def for_all_dtypes(name='dtype', no_float16=False, no_bool=False, no_complex=False):
    """Decorator that checks the fixture with all dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         no_complex(bool): If ``True``, ``numpy.complex64`` and
             ``numpy.complex128`` are omitted from candidate dtypes.

    dtypes to be tested: ``numpy.complex64`` (optional),
    ``numpy.complex128`` (optional),
    ``numpy.float16`` (optional), ``numpy.float32``,
    ``numpy.float64``, ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,
    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,
    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).

    The usage is as follows.
    This test fixture checks if ``cPickle`` successfully reconstructs
    :class:`cupy.ndarray` for various dtypes.
    ``dtype`` is an argument inserted by the decorator.

    >>> import unittest
    >>> from cupy import testing
    >>> class TestNpz(unittest.TestCase):
    ...
    ...     @testing.for_all_dtypes()
    ...     def test_pickle(self, dtype):
    ...         a = testing.shaped_arange((2, 3, 4), dtype=dtype)
    ...         s = pickle.dumps(a)
    ...         b = pickle.loads(s)
    ...         testing.assert_array_equal(a, b)

    Typically, we use this decorator in combination with
    decorators that check consistency between NumPy and CuPy like
    :func:`cupy.testing.numpy_cupy_allclose`.
    The following is such an example.

    >>> import unittest
    >>> from cupy import testing
    >>> class TestMean(unittest.TestCase):
    ...
    ...     @testing.for_all_dtypes()
    ...     @testing.numpy_cupy_allclose()
    ...     def test_mean_all(self, xp, dtype):
    ...         a = testing.shaped_arange((2, 3), xp, dtype)
    ...         return a.mean()

    .. seealso:: :func:`cupy.testing.for_dtypes`
    """
    return for_dtypes(_make_all_dtypes(no_float16, no_bool, no_complex), name=name)