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
def test_like_dtype_non_const_str(self):
    pyfunc = self.pyfunc

    @njit
    def func(n, dt):
        return pyfunc(n, dt)
    with self.assertRaises(TypingError) as raises:
        func(np.ones(4), 'int32')
    excstr = str(raises.exception)
    msg = f'If np.{self.pyfunc.__name__} dtype is a string it must be a string constant.'
    self.assertIn(msg, excstr)
    self.assertIn('{}(array(float64, 1d, C), unicode_type)'.format(pyfunc.__name__), excstr)