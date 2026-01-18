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
def test_index_exception1(self):
    pyfunc = list_index3
    cfunc = jit(nopython=True)(pyfunc)
    msg = 'arg "start" must be an Integer.'
    with self.assertRaisesRegex(errors.TypingError, msg):
        cfunc(10, 0, 'invalid', 5)