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
def test_create_nested_list(self):
    pyfunc = create_nested_list
    cfunc = njit((types.int32, types.int32, types.int32, types.int32, types.int32, types.int32))(pyfunc)
    self.assertEqual(cfunc(1, 2, 3, 4, 5, 6), pyfunc(1, 2, 3, 4, 5, 6))