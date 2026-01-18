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
def make_overload_attribute_test(self, costmodel, should_inline):
    Dummy, DummyType = self.make_dummy_type()

    @overload_attribute(DummyType, 'inlineme', inline=costmodel)
    def _get_inlineme(obj):

        def get(obj):
            return 'MY INLINED ATTRS'
        return get

    def foo(obj):
        return obj.inlineme
    self.check_getattr(test_impl=foo, args=[Dummy()], expected='MY INLINED ATTRS', block_count=1, expects_inlined=should_inline)