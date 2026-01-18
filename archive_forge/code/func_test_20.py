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
def test_20(self):

    @njit
    def foo():
        l = []
        a1 = np.arange(20)
        a2 = np.ones(5, dtype=np.complex128)
        tup = (a1, a2)
        for t in literal_unroll(tup):
            l.append(t.sum())
        return l
    self.assertEqual(foo(), foo.py_func())