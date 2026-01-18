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
def test_ndarray_oned(self):

    @njit
    def foo():
        return List(np.array(1))
    expected = List()
    expected.append(1)
    self.assertEqual(expected, foo())
    self.assertEqual(expected, foo.py_func())