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
def test_inline_operators_inplace_binop(self):

    def impl_inline(x):
        x += 1

    def impl_noinline(x):
        x -= 1
    Dummy, DummyType = self.make_dummy_type()
    dummy_inplace_binop_impl = lambda a, b: True
    setattr(Dummy, '__iadd__', dummy_inplace_binop_impl)
    setattr(Dummy, '__isub__', dummy_inplace_binop_impl)

    @overload(operator.iadd, inline='always')
    def overload_dummy_iadd(a, b):
        if isinstance(a, DummyType):
            return dummy_inplace_binop_impl

    @overload(operator.isub, inline='never')
    def overload_dummy_isub(a, b):
        if isinstance(a, DummyType):
            return dummy_inplace_binop_impl

    @overload(operator.add, inline='always')
    def overload_dummy_add(a, b):
        if isinstance(a, DummyType):
            return dummy_inplace_binop_impl

    @overload(operator.sub, inline='never')
    def overload_dummy_sub(a, b):
        if isinstance(a, DummyType):
            return dummy_inplace_binop_impl
    self.check(impl_inline, Dummy(), inline_expect={'iadd': True})
    self.check(impl_noinline, Dummy(), inline_expect={'isub': False})