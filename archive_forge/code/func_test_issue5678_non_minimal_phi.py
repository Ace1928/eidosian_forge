import sys
import copy
import logging
import numpy as np
from numba import njit, jit, types
from numba.core import errors, ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.untyped_passes import ReconstructSSA, PreserveIR
from numba.core.typed_passes import NativeLowering
from numba.extending import overload
from numba.tests.support import MemoryLeakMixin, TestCase, override_config
def test_issue5678_non_minimal_phi(self):
    from numba.core.compiler import CompilerBase, DefaultPassBuilder
    from numba.core.untyped_passes import ReconstructSSA, FunctionPass, register_pass
    phi_counter = []

    @register_pass(mutates_CFG=False, analysis_only=True)
    class CheckSSAMinimal(FunctionPass):
        _name = self.__class__.__qualname__ + '.CheckSSAMinimal'

        def __init__(self):
            super().__init__(self)

        def run_pass(self, state):
            ct = 0
            for blk in state.func_ir.blocks.values():
                ct += len(list(blk.find_exprs('phi')))
            phi_counter.append(ct)
            return True

    class CustomPipeline(CompilerBase):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(CheckSSAMinimal, ReconstructSSA)
            pm.finalize()
            return [pm]

    @njit(pipeline_class=CustomPipeline)
    def while_for(n, max_iter=1):
        a = np.empty((n, n))
        i = 0
        while i <= max_iter:
            for j in range(len(a)):
                for k in range(len(a)):
                    a[j, k] = j + k
            i += 1
        return a
    self.assertPreciseEqual(while_for(10), while_for.py_func(10))
    self.assertEqual(phi_counter, [1])