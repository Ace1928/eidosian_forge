import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def test_issue_5599_name_collision(self):

    @njit
    def f(x):
        arr = np.ones(x)
        for _ in range(2):
            val = arr * arr
            arr = arr.copy()
        return arr
    got = f(5)
    expect = f.py_func(5)
    np.testing.assert_array_equal(got, expect)