import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def gen_literal_slice_part(self, arg_val, loc, scope, stmts, equiv_set, name='static_literal_slice_part'):
    static_literal_slice_part_var = ir.Var(scope, mk_unique_var(name), loc)
    static_literal_slice_part_val = ir.Const(arg_val, loc)
    static_literal_slice_part_typ = types.IntegerLiteral(arg_val)
    stmts.append(ir.Assign(value=static_literal_slice_part_val, target=static_literal_slice_part_var, loc=loc))
    self._define(equiv_set, static_literal_slice_part_var, static_literal_slice_part_typ, static_literal_slice_part_val)
    return (static_literal_slice_part_var, static_literal_slice_part_typ)