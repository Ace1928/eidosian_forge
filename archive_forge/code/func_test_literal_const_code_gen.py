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
def test_literal_const_code_gen(self):

    def impl(x):
        _CONST1 = 'PLACEHOLDER1'
        if _CONST1:
            return 3.14159
        else:
            _CONST2 = 'PLACEHOLDER2'
        return _CONST2 + 4
    new = self._literal_const_sample_generator(impl, {1: 0, 3: 20})
    iconst = impl.__code__.co_consts
    nconst = new.__code__.co_consts
    self.assertEqual(iconst, (None, 'PLACEHOLDER1', 3.14159, 'PLACEHOLDER2', 4))
    self.assertEqual(nconst, (None, 0, 3.14159, 20, 4))
    self.assertEqual(impl(None), 3.14159)
    self.assertEqual(new(None), 24)