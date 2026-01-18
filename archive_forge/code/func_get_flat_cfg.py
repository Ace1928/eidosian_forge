import numba
from numba.tests.support import TestCase, unittest
from numba.core.registry import cpu_target
from numba.core.compiler import CompilerBase, Flags
from numba.core.compiler_machinery import PassManager
from numba.core import types, ir, bytecode, compiler, ir_utils, registry
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference,
from numba.experimental import jitclass
def get_flat_cfg(func):
    func_ir = ir_utils.compile_to_numba_ir(func, dict())
    flat_blocks = ir_utils.flatten_labels(func_ir.blocks)
    self.assertEqual(max(flat_blocks.keys()) + 1, len(func_ir.blocks))
    return ir_utils.compute_cfg_from_blocks(flat_blocks)