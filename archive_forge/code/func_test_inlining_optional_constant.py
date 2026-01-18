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
def test_inlining_optional_constant(self):

    @njit(inline='always')
    def bar(a=None, b=None):
        if b is None:
            b = 123
        return (a, b)

    def impl():
        return (bar(), bar(123), bar(b=321))
    self.check(impl, block_count='SKIP', inline_expect={'bar': True})