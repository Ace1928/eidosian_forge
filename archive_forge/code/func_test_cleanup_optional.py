import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_cleanup_optional(self):
    mem = memoryview(bytearray(b'xyz'))
    tp = types.Optional(types.MemoryView(types.byte, 1, 'C'))
    self.check_argument_cleanup(tp, mem)