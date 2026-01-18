import sys
from numba import cfunc, jit
from numba.tests.support import TestCase, captured_stderr
@cfunc(add_sig, nopython=True)
def add_nocache_usecase(x, y):
    return x + y + Z