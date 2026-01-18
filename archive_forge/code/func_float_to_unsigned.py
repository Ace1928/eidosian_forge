import numpy as np
from numba.core.errors import TypingError
from numba import njit
from numba.core import types
import struct
import unittest
def float_to_unsigned(x):
    return types.uint32(x)