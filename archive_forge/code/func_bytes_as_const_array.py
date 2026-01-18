import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def bytes_as_const_array():
    return np.frombuffer(b'foo', dtype=np.uint8)