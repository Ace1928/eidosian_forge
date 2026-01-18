import math
import unittest
from numba import jit
from numba.core import types
from numba.core.errors import TypingError, NumbaTypeError
def get_nopython_func():
    return abs