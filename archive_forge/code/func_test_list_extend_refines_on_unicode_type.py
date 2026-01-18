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
def test_list_extend_refines_on_unicode_type(self):

    @njit
    def foo(string):
        l = List()
        l.extend(string)
        return l
    for func in (foo, foo.py_func):
        for string in ('a', 'abc', '\nabc\t'):
            self.assertEqual(list(func(string)), list(string))