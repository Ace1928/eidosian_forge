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
def test_inline_operators_unary(self):

    def impl_inline(x):
        return -x

    def impl_noinline(x):
        return +x
    dummy_unary_impl = lambda x: True
    Dummy, DummyType = self.make_dummy_type()
    setattr(Dummy, '__neg__', dummy_unary_impl)
    setattr(Dummy, '__pos__', dummy_unary_impl)

    @overload(operator.neg, inline='always')
    def overload_dummy_neg(x):
        if isinstance(x, DummyType):
            return dummy_unary_impl

    @overload(operator.pos, inline='never')
    def overload_dummy_pos(x):
        if isinstance(x, DummyType):
            return dummy_unary_impl
    self.check(impl_inline, Dummy(), inline_expect={'neg': True})
    self.check(impl_noinline, Dummy(), inline_expect={'pos': False})