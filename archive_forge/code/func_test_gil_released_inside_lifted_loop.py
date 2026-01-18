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
def test_gil_released_inside_lifted_loop(self):
    """
        Test the GIL can by released by a lifted loop even though the
        surrounding code uses object mode.
        """
    cfunc = jit(f_sig, forceobj=True, nogil=True)(lifted_f)
    self.check_gil_released(cfunc)