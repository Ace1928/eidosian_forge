import sys
import numpy as np
from numba import jit, prange
from numba.core import types
from numba.tests.ctypes_usecases import c_sin
from numba.tests.support import TestCase, captured_stderr
@jit(cache=True, forceobj=True)
def add_objmode_usecase(x, y):
    object()
    return x + y + Z