from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def expect_reflection_failure(fn):

    def wrapped(self, *args, **kwargs):
        self.disable_leak_check()
        with self.assertRaises(TypeError) as raises:
            fn(self, *args, **kwargs)
        expect_msg = 'cannot reflect element of reflected container'
        self.assertIn(expect_msg, str(raises.exception))
    return wrapped