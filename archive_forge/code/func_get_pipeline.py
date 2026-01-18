from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
def get_pipeline(use_canonicaliser, use_partial_typing=False):

    class NewCompiler(CompilerBase):

        def define_pipelines(self):
            pm = PassManager('custom_pipeline')
            pm.add_pass(TranslateByteCode, 'analyzing bytecode')
            pm.add_pass(IRProcessing, 'processing IR')
            pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
            if use_partial_typing:
                pm.add_pass(PartialTypeInference, 'do partial typing')
            if use_canonicaliser:
                pm.add_pass(IterLoopCanonicalization, 'Canonicalise loops')
            pm.add_pass(SimplifyCFG, 'Simplify the CFG')
            if use_partial_typing:
                pm.add_pass(ResetTypeInfo, 'resets the type info state')
            pm.add_pass(NopythonTypeInference, 'nopython frontend')
            pm.add_pass(IRLegalization, 'ensure IR is legal')
            pm.add_pass(PreserveIR, 'save IR for later inspection')
            pm.add_pass(NativeLowering, 'native lowering')
            pm.add_pass(NoPythonBackend, 'nopython mode backend')
            pm.finalize()
            return [pm]
    return NewCompiler