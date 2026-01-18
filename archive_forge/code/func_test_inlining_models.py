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
def test_inlining_models(self):

    def s17_caller_model(expr, caller_info, callee_info):
        self.assertIsInstance(expr, ir.Expr)
        self.assertEqual(expr.op, 'call')
        return self.sentinel_17_cost_model(caller_info.func_ir)

    def s17_callee_model(expr, caller_info, callee_info):
        self.assertIsInstance(expr, ir.Expr)
        self.assertEqual(expr.op, 'call')
        return self.sentinel_17_cost_model(callee_info.func_ir)
    for caller, callee in ((10, 11), (17, 11)):

        def foo():
            return callee

        @overload(foo, inline=s17_caller_model)
        def foo_ol():

            def impl():
                return callee
            return impl

        def impl(z):
            x = z + caller
            y = foo()
            return (y + 3, x)
        self.check(impl, 10, inline_expect={'foo': caller == 17})
    for caller, callee in ((11, 17), (11, 10)):

        def bar():
            return callee

        @overload(bar, inline=s17_callee_model)
        def bar_ol():

            def impl():
                return callee
            return impl

        def impl(z):
            x = z + caller
            y = bar()
            return (y + 3, x)
        self.check(impl, 10, inline_expect={'bar': callee == 17})