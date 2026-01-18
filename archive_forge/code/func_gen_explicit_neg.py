import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def gen_explicit_neg(self, arg, arg_rel, arg_typ, size_typ, loc, scope, dsize, stmts, equiv_set):
    assert not isinstance(size_typ, int)
    explicit_neg_var = ir.Var(scope, mk_unique_var('explicit_neg'), loc)
    explicit_neg_val = ir.Expr.binop(operator.add, dsize, arg, loc=loc)
    explicit_neg_typ = types.intp
    self.calltypes[explicit_neg_val] = signature(explicit_neg_typ, size_typ, arg_typ)
    stmts.append(ir.Assign(value=explicit_neg_val, target=explicit_neg_var, loc=loc))
    self._define(equiv_set, explicit_neg_var, explicit_neg_typ, explicit_neg_val)
    return (explicit_neg_var, explicit_neg_typ)