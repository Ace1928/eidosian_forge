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
def test_bad_arrays(self):
    for pyfunc in (np_stack1, np_hstack, np_vstack, np_dstack, np_column_stack):
        cfunc = nrtjit(pyfunc)
        c = np.arange(12).reshape((4, 3))
        with self.assertTypingError() as raises:
            cfunc(c, 1, c)
        self.assertIn('expecting a non-empty tuple of arrays', str(raises.exception))