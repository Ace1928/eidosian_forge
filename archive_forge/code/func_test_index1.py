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
def test_index1(self):
    self.disable_leak_check()
    pyfunc = list_index1
    cfunc = jit(nopython=True)(pyfunc)
    for v in (0, 1, 5, 10, 99999999):
        self.check_index_result(pyfunc, cfunc, (16, v))