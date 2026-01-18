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
def test_linspace_2(self):

    def pyfunc(n, m):
        return np.linspace(n, m)
    self.check_outputs(pyfunc, [(0, 4), (1, 100), (-3.5, 2.5), (-3j, 2 + 3j), (2, 1), (1 + 0.5j, 1.5j)])