from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_nanmin_nanmax_complex_basic(self):
    pyfuncs = (array_nanmin, array_nanmax)
    for pyfunc in pyfuncs:
        cfunc = jit(nopython=True)(pyfunc)

        def check(a):
            expected = pyfunc(a)
            got = cfunc(a)
            self.assertPreciseEqual(expected, got)
        real = np.linspace(-10, 10, 40)
        real[:4] = real[-1]
        real[5:9] = np.nan
        imag = real * 2
        imag[7:12] = np.nan
        a = real - imag * 1j
        check(a)
        for _ in range(10):
            self.random.shuffle(real)
            self.random.shuffle(imag)
            a = real - imag * 1j
            a[:4] = a[-1]
            check(a)