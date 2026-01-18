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
def test_alloc_size(self):
    width = types.intp.bitwidth

    def gen_func(shape, value):
        return lambda: np.full(shape, value)
    self.check_alloc_size(gen_func(1 << width - 2, 1))
    self.check_alloc_size(gen_func((1 << width - 8, 64), 1))