import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def test_guvectorize_overwrite(self):
    with captured_stdout():
        from numba import guvectorize, float64
        import numpy as np

        @guvectorize([(float64[:], float64[:])], '()->()')
        def init_values(invals, outvals):
            invals[0] = 6.5
            outvals[0] = 4.2
        invals = np.zeros(shape=(3, 3), dtype=np.float64)
        outvals = init_values(invals)
        self.assertIsInstance(invals, np.ndarray)
        correct = np.array([[6.5, 6.5, 6.5], [6.5, 6.5, 6.5], [6.5, 6.5, 6.5]])
        np.testing.assert_array_equal(invals, correct)
        self.assertIsInstance(outvals, np.ndarray)
        correct = np.array([[4.2, 4.2, 4.2], [4.2, 4.2, 4.2], [4.2, 4.2, 4.2]])
        np.testing.assert_array_equal(outvals, correct)
        invals = np.zeros(shape=(3, 3), dtype=np.float32)
        outvals = init_values(invals)
        print(invals)
        self.assertIsInstance(invals, np.ndarray)
        correct = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(invals, correct)
        self.assertIsInstance(outvals, np.ndarray)
        correct = np.array([[4.2, 4.2, 4.2], [4.2, 4.2, 4.2], [4.2, 4.2, 4.2]])
        np.testing.assert_array_equal(outvals, correct)

        @guvectorize([(float64[:], float64[:])], '()->()', writable_args=('invals',))
        def init_values(invals, outvals):
            invals[0] = 6.5
            outvals[0] = 4.2
        invals = np.zeros(shape=(3, 3), dtype=np.float32)
        outvals = init_values(invals)
        print(invals)
        self.assertIsInstance(invals, np.ndarray)
        correct = np.array([[6.5, 6.5, 6.5], [6.5, 6.5, 6.5], [6.5, 6.5, 6.5]])
        np.testing.assert_array_equal(invals, correct)
        self.assertIsInstance(outvals, np.ndarray)
        correct = np.array([[4.2, 4.2, 4.2], [4.2, 4.2, 4.2], [4.2, 4.2, 4.2]])
        np.testing.assert_array_equal(outvals, correct)