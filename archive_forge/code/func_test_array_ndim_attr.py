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
def test_array_ndim_attr(self):

    def impl(array):
        if array.ndim == 2:
            if array.shape[1] == 2:
                return 1
        else:
            return 10
    self.assert_prune(impl, (types.Array(types.float64, 2, 'C'),), [False, None], np.zeros((2, 3)))
    self.assert_prune(impl, (types.Array(types.float64, 1, 'C'),), [True, 'both'], np.zeros((2,)))