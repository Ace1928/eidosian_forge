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
def test_double_if_else_non_literal_const(self):

    def impl(x):
        one_hundred = 100
        if x == one_hundred:
            y = 3.14159
        else:
            y = 1.61803
        return y
    self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)
    self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)