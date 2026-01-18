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
def test_2d_with_dtype(self):

    def pyfunc(arg):
        return np.array(arg, dtype=np.int32)
    cfunc = nrtjit(pyfunc)
    got = cfunc([(1, 2.5), (3, 4.5)])
    self.assertPreciseEqual(got, np.int32([[1, 2], [3, 4]]))