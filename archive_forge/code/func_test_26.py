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
def test_26(self):

    @njit
    def foo(z):
        a = (12, 12.7, 3j, 4, z, 2 * z)
        acc = 0
        count = 0
        untouched = 54
        read_only = 17
        mutated = np.empty((len(a),), dtype=np.complex128)
        for x in literal_unroll(a):
            acc += x
            mutated[count] = x
            count += 1
            escape = count + read_only
        return (escape, acc, untouched, read_only, mutated)
    f = 9
    k = f
    self.assertPreciseEqual(foo(k), foo.py_func(k))