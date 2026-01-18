from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def test_invalid_slice(self):
    self.disable_leak_check()
    pyfunc = list_getslice3
    cfunc = jit(nopython=True)(pyfunc)
    with self.assertRaises(ValueError) as cm:
        cfunc(10, 1, 2, 0)
    self.assertEqual(str(cm.exception), 'slice step cannot be zero')