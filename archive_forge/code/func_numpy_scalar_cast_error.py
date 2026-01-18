import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def numpy_scalar_cast_error():
    np.int32(np.zeros((4,)))