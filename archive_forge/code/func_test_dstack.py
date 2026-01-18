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
def test_dstack(self):
    pyfunc = np_dstack
    cfunc = nrtjit(pyfunc)
    self.check_xxstack(pyfunc, cfunc)
    a = np.arange(5)
    b = a + 10
    self.check_stack(pyfunc, cfunc, (a, b, b))
    a = np.arange(12).reshape((3, 4))
    b = a + 100
    self.check_stack(pyfunc, cfunc, (a, b, b))