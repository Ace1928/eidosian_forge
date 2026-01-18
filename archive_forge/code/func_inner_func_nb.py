import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
@jit('void(double[:], double[:], double[:])', nopython=True, nogil=True)
def inner_func_nb(result, a, b):
    """
                Function under test.
                """
    for i in range(len(result)):
        result[i] = math.exp(2.1 * a[i] + 3.2 * b[i])