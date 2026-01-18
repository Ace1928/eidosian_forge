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
def test_bad_dtype(self):

    @njit
    def func(obj, dt):
        return np.array(obj, dt)
    msg = '.*The argument "dtype" must be a data-type if it is provided.*'
    with self.assertRaisesRegex(TypingError, msg) as raises:
        func(5, 4)