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
def test_linspace_accuracy(self):

    @nrtjit
    def foo(n, m, p):
        return np.linspace(n, m, p)
    n, m, p = (0.0, 1.0, 100)
    self.assertPreciseEqual(foo(n, m, p), foo.py_func(n, m, p))