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
def test_issue5482_objmode_expr_null_lowering(self):
    from numba.core.compiler import CompilerBase, DefaultPassBuilder
    from numba.core.untyped_passes import ReconstructSSA, IRProcessing
    from numba.core.typed_passes import PreLowerStripPhis

    class CustomPipeline(CompilerBase):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_objectmode_pipeline(self.state)
            pm.add_pass_after(ReconstructSSA, IRProcessing)
            pm.add_pass_after(PreLowerStripPhis, ReconstructSSA)
            pm.finalize()
            return [pm]

    @jit('(intp, intp, intp)', looplift=False, pipeline_class=CustomPipeline)
    def foo(x, v, n):
        for i in range(n):
            if i == n:
                if i == x:
                    pass
                else:
                    problematic = v
            elif i == x:
                pass
            else:
                problematic = problematic + v
        return problematic