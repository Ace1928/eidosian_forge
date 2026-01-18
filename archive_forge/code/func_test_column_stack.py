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
def test_column_stack(self):
    pyfunc = np_column_stack
    cfunc = nrtjit(pyfunc)
    a = np.arange(4)
    b = a + 10
    c = np.arange(12).reshape((4, 3))
    self.check_stack(pyfunc, cfunc, (a, b, c))
    self.assert_no_memory_leak()
    self.disable_leak_check()
    a = np.array(42)
    with self.assertTypingError():
        cfunc((a, a, a))
    a = a.reshape((1, 1, 1))
    with self.assertTypingError():
        cfunc((a, a, a))