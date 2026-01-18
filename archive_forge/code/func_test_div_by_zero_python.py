from numba import jit
import unittest
def test_div_by_zero_python(self):

    @jit
    def model_python(val):
        return 1 / val
    with self.assertRaises(ZeroDivisionError):
        model_python(0)