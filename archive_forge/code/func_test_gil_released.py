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
def test_gil_released(self):
    """
        Test releasing the GIL, by checking parallel runs produce
        unpredictable results.
        """
    cfunc = jit(f_sig, nopython=True, nogil=True)(f)
    self.check_gil_released(cfunc)