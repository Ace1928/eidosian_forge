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
@skip_if_32bit
def test_invalid_size_array(self):

    @njit
    def foo(x):
        np.empty(x)
    self.disable_leak_check()
    with self.assertRaises(MemoryError) as raises:
        foo(types.size_t.maxval // 8 // 2)
    self.assertIn('Allocation failed', str(raises.exception))