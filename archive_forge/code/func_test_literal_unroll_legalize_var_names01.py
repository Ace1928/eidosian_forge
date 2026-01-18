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
def test_literal_unroll_legalize_var_names01(self):
    test = np.array([(1, 2), (2, 3)], dtype=[('a1', 'f8'), ('a2', 'f8')])
    fields = tuple(test.dtype.fields.keys())

    @njit
    def foo(arr):
        res = 0
        for k in literal_unroll(fields):
            res = res + np.abs(arr[k]).sum()
        return res
    self.assertEqual(foo(test), 8.0)