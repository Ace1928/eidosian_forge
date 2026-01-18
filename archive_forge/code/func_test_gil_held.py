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
def test_gil_held(self):
    """
        Test the GIL is held by default, by checking serialized runs
        produce deterministic results.
        """
    cfunc = jit(f_sig, nopython=True)(f)
    self.check_gil_held(cfunc)