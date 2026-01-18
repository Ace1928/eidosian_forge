import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
def test_inline_always_ssa_scope_validity(self):

    def bar():
        b = 5
        while b > 1:
            b //= 2
        return 10

    @overload(bar, inline='always')
    def bar_impl():
        return bar

    @njit
    def foo():
        bar()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', errors.NumbaIRAssumptionWarning)
        ignore_internal_warnings()
        self.assertEqual(foo(), foo.py_func())
    self.assertEqual(len(w), 0)