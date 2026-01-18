import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def real_np_types():
    for tp_name in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'intc', 'uintc', 'intp', 'uintp', 'float32', 'float64', 'bool_'):
        yield tp_name