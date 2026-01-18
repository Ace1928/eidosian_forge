import ctypes
import ctypes.util
import os
import sys
import threading
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core import errors
from numba.tests.support import TestCase, tag
def test_gil_released_by_caller_and_callee(self):
    """
        Same, but with both caller and callee asking to release the GIL.
        """
    compiled_f = jit(f_sig, nopython=True, nogil=True)(f)

    @jit(f_sig, nopython=True, nogil=True)
    def caller(a, i):
        compiled_f(a, i)
    self.check_gil_released(caller)