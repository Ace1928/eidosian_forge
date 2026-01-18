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
def test_issue5824(self):
    """ Similar to the above test_issue5792, checks mutation of the inlinee
        IR is local only"""

    class CustomCompiler(CompilerBase):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(InlineOverloads, InlineOverloads)
            pm.finalize()
            return [pm]

    def bar(x):
        ...

    @overload(bar, inline='always')
    def ol_bar(x):
        if isinstance(x, types.Integer):

            def impl(x):
                return x + 1.3
            return impl

    @njit(pipeline_class=CustomCompiler)
    def foo(z):
        return (bar(z), bar(z))
    self.assertEqual(foo(10), (11.3, 11.3))