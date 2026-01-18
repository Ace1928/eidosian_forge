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
def test_2d_dtype_from_type(self):

    def func(m, n):
        return np.full((m, n), np.int32(1))
    self.check_2d(func)

    def func(m, n):
        return np.full((m, n), np.complex128(1))
    self.check_2d(func)

    def func(m, n):
        return np.full((m, n), 1, dtype=np.int8)
    self.check_2d(func)