import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_exception_on_too_many_args(self):

    @njit
    def foo():
        l = List((0, 1, 2), (3, 4, 5))
        return l
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('List() expected at most 1 argument, got 2', str(raises.exception))
    with self.assertRaises(TypeError) as raises:
        List((0, 1, 2), (3, 4, 5))
    self.assertIn('List() expected at most 1 argument, got 2', str(raises.exception))

    @njit
    def foo():
        l = List((0, 1, 2), (3, 4, 5), (6, 7, 8))
        return l
    with self.assertRaises(TypingError) as raises:
        foo()
    self.assertIn('List() expected at most 1 argument, got 3', str(raises.exception))
    with self.assertRaises(TypeError) as raises:
        List((0, 1, 2), (3, 4, 5), (6, 7, 8))
    self.assertIn('List() expected at most 1 argument, got 3', str(raises.exception))