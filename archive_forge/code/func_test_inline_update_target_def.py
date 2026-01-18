import re
import numpy as np
from numba.tests.support import (TestCase, override_config, captured_stdout,
from numba import jit, njit
from numba.core import types, ir, postproc, compiler
from numba.core.ir_utils import (guard, find_callname, find_const,
from numba.core.registry import CPUDispatcher
from numba.core.inline_closurecall import inline_closure_call
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
import unittest
@skip_parfors_unsupported
def test_inline_update_target_def(self):

    def test_impl(a):
        if a == 1:
            b = 2
        else:
            b = 3
        return b
    func_ir = compiler.run_frontend(test_impl)
    blocks = list(func_ir.blocks.values())
    for block in blocks:
        for i, stmt in enumerate(block.body):
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var) and (guard(find_const, func_ir, stmt.value) == 2):
                func_ir._definitions[stmt.target.name].remove(stmt.value)
                stmt.value = ir.Expr.call(ir.Var(block.scope, 'myvar', loc=stmt.loc), (), (), stmt.loc)
                func_ir._definitions[stmt.target.name].append(stmt.value)
                inline_closure_call(func_ir, {}, block, i, lambda: 2)
                break
    self.assertEqual(len(func_ir._definitions['b']), 2)