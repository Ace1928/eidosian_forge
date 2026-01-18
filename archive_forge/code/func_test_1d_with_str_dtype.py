import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_1d_with_str_dtype(self):

    def pyfunc(arg):
        return np.array(arg, dtype='float32')
    self.check_outputs(pyfunc, [([2, 42],), ([3.5, 1.0],), ((1, 3.5, 42),), ((),)])