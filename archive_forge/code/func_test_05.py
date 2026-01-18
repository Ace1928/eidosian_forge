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
def test_05(self):

    @njit
    def foo(tup1, tup2):
        acc = 0
        for a in literal_unroll(tup1):
            if a[0] > 1:
                acc += tup2[0].sum()
        return acc
    n = 10
    tup1 = [np.zeros(10), np.zeros(10)]
    tup2 = (np.ones((n,)), np.ones((n, n)), np.ones((n, n, n)), np.ones((n, n, n, n)), np.ones((n, n, n, n, n)))
    with self.assertRaises(errors.UnsupportedError) as raises:
        foo(tup1, tup2)
    msg = 'Invalid use of literal_unroll with a function argument'
    self.assertIn(msg, str(raises.exception))