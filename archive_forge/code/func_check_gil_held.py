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
def check_gil_held(self, func):
    arr = self.run_in_threads(func, n_threads=4)
    distinct = set(arr)
    self.assertEqual(len(distinct), 1, distinct)