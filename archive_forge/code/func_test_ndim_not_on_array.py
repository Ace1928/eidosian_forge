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
def test_ndim_not_on_array(self):
    FakeArray = collections.namedtuple('FakeArray', ['ndim'])
    fa = FakeArray(ndim=2)

    def impl(fa):
        if fa.ndim == 2:
            return fa.ndim
        else:
            object()
    self.assert_prune(impl, (types.Array(types.float64, 2, 'C'),), [False], np.zeros((2, 3)))
    FakeArrayType = types.NamedUniTuple(types.int64, 1, FakeArray)
    self.assert_prune(impl, (FakeArrayType,), [None], fa, flags={'nopython': False, 'forceobj': True})