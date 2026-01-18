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
def test_1d_dtype_invalid_str(self):

    @njit
    def func(n, fv):
        return np.full(n, fv, 'ABCDEF')
    with self.assertRaises(TypingError) as raises:
        func((5,), 4.5)
    excstr = str(raises.exception)
    self.assertIn("Invalid NumPy dtype specified: 'ABCDEF'", excstr)