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
def test_0d(self):
    """
        stack(0d arrays)
        """
    pyfunc = np_stack1
    cfunc = nrtjit(pyfunc)
    a = np.array(42)
    b = np.array(-5j)
    c = np.array(True)
    self.check_stack(pyfunc, cfunc, (a, b, c))