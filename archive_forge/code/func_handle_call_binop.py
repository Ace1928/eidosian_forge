import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def handle_call_binop(cond_def):
    br = None
    if cond_def.fn == operator.eq:
        br = inst.truebr
        otherbr = inst.falsebr
        cond_val = 1
    elif cond_def.fn == operator.ne:
        br = inst.falsebr
        otherbr = inst.truebr
        cond_val = 0
    lhs_typ = self.typemap[cond_def.lhs.name]
    rhs_typ = self.typemap[cond_def.rhs.name]
    if br is not None and (isinstance(lhs_typ, types.Integer) and isinstance(rhs_typ, types.Integer) or (isinstance(lhs_typ, types.BaseTuple) and isinstance(rhs_typ, types.BaseTuple))):
        loc = inst.loc
        args = (cond_def.lhs, cond_def.rhs)
        asserts = self._make_assert_equiv(scope, loc, equiv_set, args)
        asserts.append(ir.Assign(ir.Const(cond_val, loc), cond_var, loc))
        self.prepends[label, br] = asserts
        self.prepends[label, otherbr] = [ir.Assign(ir.Const(1 - cond_val, loc), cond_var, loc)]