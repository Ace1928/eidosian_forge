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
def test_double_if_else_rt_const(self):

    def impl(x):
        one_hundred = 100
        x_is_none_work = 4
        if x is None:
            x_is_none_work = 100
        else:
            dead = 7
        if x_is_none_work == one_hundred:
            y = 10
        else:
            y = -3
        return (y, x_is_none_work)
    self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
    self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)