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
def test_freevar_bindings(self):

    def impl():
        x = foo()
        y = bar()
        z = baz()
        return (x, y, z)
    opts = ('always', 'never')
    for inline_foo, inline_bar, inline_baz in product(opts, opts, opts):

        def foo():
            x = 10
            y = 20
            z = x + 12
            return (x, y + 3, z)

        def bar():
            x = 30
            y = 40
            z = x + 12
            return (x, y + 3, z)

        def baz():
            x = 60
            y = 80
            z = x + 12
            return (x, y + 3, z)

        def factory(target, x, y, inline=None):
            z = x + 12

            @overload(target, inline=inline)
            def func():

                def impl():
                    return (x, y + 3, z)
                return impl
        factory(foo, 10, 20, inline=inline_foo)
        factory(bar, 30, 40, inline=inline_bar)
        factory(baz, 60, 80, inline=inline_baz)
        inline_expect = {'foo': self.inline_opt_as_bool[inline_foo], 'bar': self.inline_opt_as_bool[inline_bar], 'baz': self.inline_opt_as_bool[inline_baz]}
        self.check(impl, inline_expect=inline_expect)