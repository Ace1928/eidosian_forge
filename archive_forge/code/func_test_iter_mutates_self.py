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
def test_iter_mutates_self(self):
    self.disable_leak_check()

    @njit
    def foo(x):
        count = 0
        for i in x:
            if count > 1:
                x.append(2.0)
            count += 1
    l = List()
    l.append(1.0)
    l.append(1.0)
    l.append(1.0)
    with self.assertRaises(RuntimeError) as raises:
        foo(l)
    msg = 'list was mutated during iteration'
    self.assertIn(msg, str(raises.exception))