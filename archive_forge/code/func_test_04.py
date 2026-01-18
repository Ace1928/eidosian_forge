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
def test_04(self):

    @njit
    def foo():
        x = [12, 12.7, 3j, 4]
        y = ('foo', 8)
        acc = 0
        for a in literal_unroll(x):
            acc += a
            if acc.real < 26:
                acc -= 1
            else:
                for t in literal_unroll(y):
                    acc += t is False
                break
        return acc
    self.assertEqual(foo(), foo.py_func())