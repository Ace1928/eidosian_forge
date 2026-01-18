import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def func_np(a, b):
    """
                Control function using Numpy.
                """
    return np.exp(2.1 * a + 3.2 * b)