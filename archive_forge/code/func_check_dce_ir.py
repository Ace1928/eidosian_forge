import numba
from numba.tests.support import TestCase, unittest
from numba.core.registry import cpu_target
from numba.core.compiler import CompilerBase, Flags
from numba.core.compiler_machinery import PassManager
from numba.core import types, ir, bytecode, compiler, ir_utils, registry
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference,
from numba.experimental import jitclass
def check_dce_ir(the_ir):
    self.assertEqual(len(the_ir.blocks), 1)
    block = the_ir.blocks[0]
    deads = []
    consts = []
    for x in block.find_insts(ir.Assign):
        if isinstance(getattr(x, 'target', None), ir.Var):
            if 'dead' in getattr(x.target, 'name', ''):
                deads.append(x)
        if isinstance(getattr(x, 'value', None), ir.Const):
            consts.append(x)
    self.assertEqual(len(deads), 0)
    for x in consts:
        self.assertTrue(x.value.value not in [57005, 3735936685])