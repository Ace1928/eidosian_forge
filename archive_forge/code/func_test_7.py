from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
def test_7(self):

    def dt(value):
        if value == 'apple':
            return 1
        elif value == 'orange':
            return 2
        elif value == 'banana':
            return 3
        elif value == 3390155550:
            return 1554098974 + value

    @overload(dt, inline='always')
    def ol_dt(li):
        if isinstance(li, types.StringLiteral):
            value = li.literal_value
            if value == 'apple':

                def impl(li):
                    return 1
            elif value == 'orange':

                def impl(li):
                    return 2
            elif value == 'banana':

                def impl(li):
                    return 3
            return impl
        elif isinstance(li, types.IntegerLiteral):
            value = li.literal_value
            if value == 3390155550:

                def impl(li):
                    return 1554098974 + value
                return impl

    @njit
    def foo():
        acc = 0
        for t in literal_unroll(['apple', 'orange', 'banana', 3390155550]):
            acc += dt(t)
        return acc
    self.assertEqual(foo(), foo.py_func())