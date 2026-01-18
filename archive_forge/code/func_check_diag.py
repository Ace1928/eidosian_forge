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
def check_diag(self, pyfunc, nrtfunc, *args, **kwargs):
    expected = pyfunc(*args, **kwargs)
    computed = nrtfunc(*args, **kwargs)
    self.assertEqual(computed.size, expected.size)
    self.assertEqual(computed.dtype, expected.dtype)
    np.testing.assert_equal(expected, computed)