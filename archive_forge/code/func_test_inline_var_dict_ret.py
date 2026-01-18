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
def test_inline_var_dict_ret(self):

    @njit(locals={'b': types.float64})
    def g(a):
        b = a + 1
        return b

    def test_impl():
        return g(1)
    func_ir = compiler.run_frontend(test_impl)
    blocks = list(func_ir.blocks.values())
    for block in blocks:
        for i, stmt in enumerate(block.body):
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'call'):
                func_def = guard(get_definition, func_ir, stmt.value.func)
                if isinstance(func_def, (ir.Global, ir.FreeVar)) and isinstance(func_def.value, CPUDispatcher):
                    py_func = func_def.value.py_func
                    _, var_map = inline_closure_call(func_ir, py_func.__globals__, block, i, py_func)
                    break
    self.assertTrue('b' in var_map)