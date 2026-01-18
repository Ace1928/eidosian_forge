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
def test_15(self):

    @njit
    def foo(x):
        acc = 0
        for a in literal_unroll(x):
            acc += len(a)
        return a
    n = 5
    tup = (np.ones((n,)), np.ones((n, n)), 'ABCDEFGHJI', (1, 2, 3), (1, 'foo', 2, 'bar'), {3, 4, 5, 6, 7})
    with self.assertRaises(errors.TypingError) as raises:
        foo(tup)
    self.assertIn('Cannot unify', str(raises.exception))