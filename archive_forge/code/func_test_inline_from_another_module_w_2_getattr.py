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
def test_inline_from_another_module_w_2_getattr(self):
    import numba.tests.inlining_usecases
    import numba.tests as nt

    def impl():
        z = _GLOBAL1 + 2
        return (nt.inlining_usecases.baz(), z)
    self.check(impl, inline_expect={'baz': True})