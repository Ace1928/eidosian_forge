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
def test_basic_inline_never(self):

    def foo():
        pass

    @overload(foo, inline='never')
    def foo_overload():

        def foo_impl():
            pass
        return foo_impl

    def impl():
        return foo()
    self.check(impl, inline_expect={'foo': False})