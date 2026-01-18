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
def test_25(self):

    @njit
    def foo():
        val = literal_unroll(((1, 2, 3), (2j, 3j), [1, 2], 'xyz'))
        alias1 = val
        alias2 = alias1
        lens = []
        for x in alias2:
            lens.append(len(x))
        return lens
    self.assertEqual(foo(), foo.py_func())