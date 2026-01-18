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
def test_extend_other(self):

    @njit
    def impl(other):
        l = List.empty_list(types.int32)
        for x in range(10):
            l.append(x)
        l.extend(other)
        return l
    other = List.empty_list(types.int32)
    for x in range(10):
        other.append(x)
    expected = impl.py_func(other)
    got = impl(other)
    self.assertEqual(expected, got)