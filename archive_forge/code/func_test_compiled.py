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
def test_compiled(self):

    @njit
    def producer():
        l = List.empty_list(int32)
        l.append(23)
        return l

    @njit
    def consumer(l):
        return l[0]
    l = producer()
    val = consumer(l)
    self.assertEqual(val, 23)