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
def test_ctype_wrapping(self):
    pyfunc = use_ctype_wrapping
    cfunc = njit((types.double,))(pyfunc)
    x = 3.14
    self.assertEqual(pyfunc(x), cfunc(x))