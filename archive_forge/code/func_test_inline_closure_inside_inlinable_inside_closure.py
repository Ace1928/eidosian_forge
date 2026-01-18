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
def test_inline_closure_inside_inlinable_inside_closure(self):

    @njit(inline='always')
    def foo(a):

        def baz():
            return 12 + a
        return baz() + 8

    def impl():
        z = 9

        def bar(x):
            return foo(z) + 7 + x
        return bar(z + 2)
    self.check(impl, inline_expect={'foo': True}, block_count=1)