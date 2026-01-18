import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def run_untyped_passes(self, func, enable_ssa=False):
    """
        Run the compiler frontend's untyped passes over the given Python
        function, and return the function's canonical Numba IR.

        Disable SSA transformation by default, since the call site won't be in
        SSA form and self.inline_ir depends on this being the case.
        """
    from numba.core.compiler import StateDict, _CompileStatus
    from numba.core.untyped_passes import ExtractByteCode
    from numba.core import bytecode
    from numba.parfors.parfor import ParforDiagnostics
    state = StateDict()
    state.func_ir = None
    state.typingctx = self.typingctx
    state.targetctx = self.targetctx
    state.locals = self.locals
    state.pipeline = self.pipeline
    state.flags = self.flags
    state.flags.enable_ssa = enable_ssa
    state.func_id = bytecode.FunctionIdentity.from_function(func)
    state.typemap = None
    state.calltypes = None
    state.type_annotation = None
    state.status = _CompileStatus(False)
    state.return_type = None
    state.parfor_diagnostics = ParforDiagnostics()
    state.metadata = {}
    ExtractByteCode().run_pass(state)
    state.args = len(state.bc.func_id.pysig.parameters) * (types.pyobject,)
    pm = self._compiler_pipeline(state)
    pm.finalize()
    pm.run(state)
    return state.func_ir