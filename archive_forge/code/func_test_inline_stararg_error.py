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
def test_inline_stararg_error(self):

    def foo(a, *b):
        return a + b[0]

    @overload(foo, inline='always')
    def overload_foo(a, *b):
        return lambda a, *b: a + b[0]

    def impl():
        return foo(3, 3, 5)
    with self.assertRaises(NotImplementedError) as e:
        self.check(impl, inline_expect={'foo': True})
    self.assertIn('Stararg not supported in inliner for arg 1 *b', str(e.exception))