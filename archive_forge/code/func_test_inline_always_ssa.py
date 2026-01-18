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
def test_inline_always_ssa(self):
    dummy_true = True

    def foo(A):
        return True

    @overload(foo, inline='always')
    def foo_overload(A):

        def impl(A):
            s = dummy_true
            for i in range(len(A)):
                dummy = dummy_true
                if A[i]:
                    dummy = A[i]
                s *= dummy
            return s
        return impl

    def impl():
        return foo(np.array([True, False, True]))
    self.check(impl, block_count='SKIP', inline_expect={'foo': True})