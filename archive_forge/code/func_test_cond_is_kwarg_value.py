import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
def test_cond_is_kwarg_value(self):

    def impl(x=1000):
        if x == 1000:
            y = 10
        else:
            y = 40
        if x != 1000:
            z = 100
        else:
            z = 400
        return (z, y)
    self.assert_prune(impl, (types.Omitted(1000),), [None, None], 1000)
    self.assert_prune(impl, (types.IntegerLiteral(1000),), [None, None], 1000)
    self.assert_prune(impl, (types.IntegerLiteral(0),), [None, None], 0)
    self.assert_prune(impl, (types.NoneType('none'),), [True, False], None)