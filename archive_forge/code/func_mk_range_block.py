import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def mk_range_block(typemap, start, stop, step, calltypes, scope, loc):
    """make a block that initializes loop range and iteration variables.
    target label in jump needs to be set.
    """
    g_range_var = ir.Var(scope, mk_unique_var('$range_g_var'), loc)
    typemap[g_range_var.name] = get_global_func_typ(range)
    g_range = ir.Global('range', range, loc)
    g_range_assign = ir.Assign(g_range, g_range_var, loc)
    arg_nodes, args = _mk_range_args(typemap, start, stop, step, scope, loc)
    range_call = ir.Expr.call(g_range_var, args, (), loc)
    calltypes[range_call] = typemap[g_range_var.name].get_call_type(typing.Context(), [types.intp] * len(args), {})
    range_call_var = ir.Var(scope, mk_unique_var('$range_c_var'), loc)
    typemap[range_call_var.name] = types.iterators.RangeType(types.intp)
    range_call_assign = ir.Assign(range_call, range_call_var, loc)
    iter_call = ir.Expr.getiter(range_call_var, loc)
    calltypes[iter_call] = signature(types.range_iter64_type, types.range_state64_type)
    iter_var = ir.Var(scope, mk_unique_var('$iter_var'), loc)
    typemap[iter_var.name] = types.iterators.RangeIteratorType(types.intp)
    iter_call_assign = ir.Assign(iter_call, iter_var, loc)
    phi_var = ir.Var(scope, mk_unique_var('$phi'), loc)
    typemap[phi_var.name] = types.iterators.RangeIteratorType(types.intp)
    phi_assign = ir.Assign(iter_var, phi_var, loc)
    jump_header = ir.Jump(-1, loc)
    range_block = ir.Block(scope, loc)
    range_block.body = arg_nodes + [g_range_assign, range_call_assign, iter_call_assign, phi_assign, jump_header]
    return range_block