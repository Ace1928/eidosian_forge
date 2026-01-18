import unittest
from numba.tests.support import captured_stdout, skip_parfors_unsupported
from numba import set_parallel_chunksize
from numba.tests.support import TestCase
def test_chunksize_with(self):
    with captured_stdout():
        from numba import njit, prange, parallel_chunksize

        @njit(parallel=True)
        def func1(n):
            acc = 0
            for i in prange(n):
                acc += i
            return acc

        @njit(parallel=True)
        def func2(n):
            acc = 0
            with parallel_chunksize(8):
                for i in prange(n):
                    acc += i
            return acc
        with parallel_chunksize(4):
            result1 = func1(12)
            result2 = func2(12)
            result3 = func1(12)
        self.assertPreciseEqual(result1, func1.py_func(12))
        self.assertPreciseEqual(result2, func2.py_func(12))
        self.assertPreciseEqual(result3, func1.py_func(12))