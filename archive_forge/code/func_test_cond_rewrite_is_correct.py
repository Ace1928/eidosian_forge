import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
def test_cond_rewrite_is_correct(self):

    def fn(x):
        if x is None:
            return 10
        return 12

    def check(func, arg_tys, bit_val):
        func_ir = compile_to_ir(func)
        before_branches = self.find_branches(func_ir)
        self.assertEqual(len(before_branches), 1)
        pred_var = before_branches[0].cond
        pred_defn = ir_utils.get_definition(func_ir, pred_var)
        self.assertEqual(pred_defn.op, 'call')
        condition_var = pred_defn.args[0]
        condition_op = ir_utils.get_definition(func_ir, condition_var)
        self.assertEqual(condition_op.op, 'binop')
        if self._DEBUG:
            print('=' * 80)
            print('before prune')
            func_ir.dump()
        dead_branch_prune(func_ir, arg_tys)
        if self._DEBUG:
            print('=' * 80)
            print('after prune')
            func_ir.dump()
        new_condition_defn = ir_utils.get_definition(func_ir, condition_var)
        self.assertTrue(isinstance(new_condition_defn, ir.Const))
        self.assertEqual(new_condition_defn.value, bit_val)
    check(fn, (types.NoneType('none'),), 1)
    check(fn, (types.IntegerLiteral(10),), 0)