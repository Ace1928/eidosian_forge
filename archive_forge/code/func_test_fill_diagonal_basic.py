from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_fill_diagonal_basic(self):
    pyfunc = numpy_fill_diagonal
    cfunc = jit(nopython=True)(pyfunc)

    def _shape_variations(n):
        yield (n, n)
        yield (2 * n, n)
        yield (n, 2 * n)
        yield (2 * n + 1, 2 * n - 1)
        yield (n, n, n, n)
        yield (1, 1, 1)

    def _val_variations():
        yield 1
        yield 3.142
        yield np.nan
        yield (-np.inf)
        yield True
        yield np.arange(4)
        yield (4,)
        yield [8, 9]
        yield np.arange(54).reshape(9, 3, 2, 1)
        yield np.asfortranarray(np.arange(9).reshape(3, 3))
        yield np.arange(9).reshape(3, 3)[::-1]

    def _multi_dimensional_array_variations(n):
        for shape in _shape_variations(n):
            yield np.zeros(shape, dtype=np.float64)
            yield np.asfortranarray(np.ones(shape, dtype=np.float64))

    def _multi_dimensional_array_variations_strided(n):
        for shape in _shape_variations(n):
            tmp = np.zeros(tuple([x * 2 for x in shape]), dtype=np.float64)
            slicer = tuple((slice(0, x * 2, 2) for x in shape))
            yield tmp[slicer]

    def _check_fill_diagonal(arr, val):
        for wrap in (None, True, False):
            a = arr.copy()
            b = arr.copy()
            if wrap is None:
                params = {}
            else:
                params = {'wrap': wrap}
            pyfunc(a, val, **params)
            cfunc(b, val, **params)
            self.assertPreciseEqual(a, b)
    for arr in _multi_dimensional_array_variations(3):
        for val in _val_variations():
            _check_fill_diagonal(arr, val)
    for arr in _multi_dimensional_array_variations_strided(3):
        for val in _val_variations():
            _check_fill_diagonal(arr, val)
    arr = np.array([True] * 9).reshape(3, 3)
    _check_fill_diagonal(arr, False)
    _check_fill_diagonal(arr, [False, True, False])
    _check_fill_diagonal(arr, np.array([True, False, True]))