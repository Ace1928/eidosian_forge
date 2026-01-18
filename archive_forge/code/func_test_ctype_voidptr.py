from ctypes import *
import sys
import threading
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.core.typing import ctypes_utils
from numba.tests.support import MemoryLeakMixin, tag, TestCase
from numba.tests.ctypes_usecases import *
import unittest
def test_ctype_voidptr(self):
    pyfunc = use_c_pointer
    cfunc = njit((types.int32,))(pyfunc)
    x = 123
    self.assertEqual(cfunc(x), x + 1)