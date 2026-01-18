import numba
from numba.tests.support import TestCase, unittest
from numba.core.registry import cpu_target
from numba.core.compiler import CompilerBase, Flags
from numba.core.compiler_machinery import PassManager
from numba.core import types, ir, bytecode, compiler, ir_utils, registry
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference,
from numba.experimental import jitclass
def test_find_const_global(self):
    """
        Test find_const() for values in globals (ir.Global) and freevars
        (ir.FreeVar) that are considered constants for compilation.
        """
    FREEVAR_C = 12

    def foo(a):
        b = GLOBAL_B
        c = FREEVAR_C
        return a + b + c
    f_ir = compiler.run_frontend(foo)
    block = f_ir.blocks[0]
    const_b = None
    const_c = None
    for inst in block.body:
        if isinstance(inst, ir.Assign) and inst.target.name == 'b':
            const_b = ir_utils.guard(ir_utils.find_const, f_ir, inst.target)
        if isinstance(inst, ir.Assign) and inst.target.name == 'c':
            const_c = ir_utils.guard(ir_utils.find_const, f_ir, inst.target)
    self.assertEqual(const_b, GLOBAL_B)
    self.assertEqual(const_c, FREEVAR_C)