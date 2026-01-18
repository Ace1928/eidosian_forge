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
def test_inline_inside_loop(self):

    @njit(inline='always')
    def foo():
        return 12

    def impl():
        acc = 0.0
        for i in range(5):
            acc += foo()
        return acc
    self.check(impl, inline_expect={'foo': True}, block_count=4)