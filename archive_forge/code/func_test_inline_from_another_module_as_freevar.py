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
def test_inline_from_another_module_as_freevar(self):

    def factory():
        from .inlining_usecases import baz

        @njit(inline='always')
        def tmp():
            return baz()
        return tmp
    bop = factory()

    def impl():
        z = _GLOBAL1 + 2
        return (bop(), z)
    self.check(impl, inline_expect={'baz': True})