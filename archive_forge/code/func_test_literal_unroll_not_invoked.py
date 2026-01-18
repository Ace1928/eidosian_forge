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
def test_literal_unroll_not_invoked(self):

    @njit(pipeline_class=CapturingCompiler)
    def foo():
        acc = 0
        for i in (1, 2, 3):
            acc += i
        return acc
    foo()
    cres = foo.overloads[foo.signatures[0]]
    self.assertFalse(cres.metadata['mutation_results'][LiteralUnroll])