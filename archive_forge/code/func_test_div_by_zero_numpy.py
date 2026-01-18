from numba import jit
import unittest
def test_div_by_zero_numpy(self):

    @jit(error_model='numpy')
    def model_numpy(val):
        return 1 / val
    self.assertEqual(model_numpy(0), float('inf'))