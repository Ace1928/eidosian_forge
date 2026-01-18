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
def test_issue9242_use_not_dom_def(self):
    from numba.core.ir import FunctionIR
    from numba.core.compiler_machinery import AnalysisPass, register_pass

    def check(fir: FunctionIR):
        [blk, *_] = fir.blocks.values()
        var = blk.scope.get('d')
        defn = fir.get_definition(var)
        self.assertEqual(defn.op, 'phi')
        self.assertIn(ir.UNDEFINED, defn.incoming_values)

    @register_pass(mutates_CFG=False, analysis_only=True)
    class SSACheck(AnalysisPass):
        """
            Check SSA on variable `d`
            """
        _name = 'SSA_Check'

        def __init__(self):
            AnalysisPass.__init__(self)

        def run_pass(self, state):
            check(state.func_ir)
            return False

    class SSACheckPipeline(CompilerBase):
        """Inject SSACheck pass into the default pipeline following the SSA
            pass
            """

        def define_pipelines(self):
            pipeline = DefaultPassBuilder.define_nopython_pipeline(self.state, 'ssa_check_custom_pipeline')
            pipeline._finalized = False
            pipeline.add_pass_after(SSACheck, ReconstructSSA)
            pipeline.finalize()
            return [pipeline]

    @njit(pipeline_class=SSACheckPipeline)
    def py_func(a):
        c = a > 0
        if c:
            d = a + 5
        return c and d > 0
    py_func(10)