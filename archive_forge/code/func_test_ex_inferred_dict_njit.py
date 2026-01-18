import unittest
from numba.tests.support import captured_stdout
def test_ex_inferred_dict_njit(self):
    with captured_stdout():
        from numba import njit
        import numpy as np

        @njit
        def foo():
            d = dict()
            k = {1: np.arange(1), 2: np.arange(2)}
            d[3] = np.arange(3)
            d[5] = np.arange(5)
            return (d, k)
        d, k = foo()
        print(d)
        print(k)
    np.testing.assert_array_equal(d[3], [0, 1, 2])
    np.testing.assert_array_equal(d[5], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(k[1], [0])
    np.testing.assert_array_equal(k[2], [0, 1])