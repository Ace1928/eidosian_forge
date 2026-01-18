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
def test_from_ctypes(self):
    """
        Test converting a ctypes type to a Numba type.
        """

    def check(cty, ty):
        got = ctypes_utils.from_ctypes(cty)
        self.assertEqual(got, ty)
    self._conversion_tests(check)
    with self.assertRaises(TypeError) as raises:
        ctypes_utils.from_ctypes(c_wchar_p)
    self.assertIn('Unsupported ctypes type', str(raises.exception))