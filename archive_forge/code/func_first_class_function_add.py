import sys
import numpy as np
from numba import jit, prange
from numba.core import types
from numba.tests.ctypes_usecases import c_sin
from numba.tests.support import TestCase, captured_stderr
@jit(cache=True)
def first_class_function_add(x):
    return x + x