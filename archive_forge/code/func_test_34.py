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
def test_34(self):

    @njit
    def foo():
        acc = 0
        l1 = [1, 2, 3, 4]
        l2 = [10, 20]
        if acc - 2 > 3:
            tup = (l1, l2)
        else:
            a1 = np.arange(20)
            a2 = np.ones(5, dtype=np.complex128)
            tup = (l1, a1, l2, a2)
        for t in literal_unroll(tup):
            acc += len(t)
        return acc
    with self.assertRaises(errors.UnsupportedError) as raises:
        foo()
    self.assertIn('Invalid use of', str(raises.exception))
    self.assertIn('found multiple definitions of variable', str(raises.exception))