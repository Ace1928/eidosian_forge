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
def gen_np_call(func_as_str, func, lhs, args, typingctx, typemap, calltypes):
    scope = args[0].scope
    loc = args[0].loc
    g_np_var = ir.Var(scope, mk_unique_var('$np_g_var'), loc)
    typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global('np', numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    np_attr_call = ir.Expr.getattr(g_np_var, func_as_str, loc)
    attr_var = ir.Var(scope, mk_unique_var('$np_attr_attr'), loc)
    func_var_typ = get_np_ufunc_typ(func)
    typemap[attr_var.name] = func_var_typ
    attr_assign = ir.Assign(np_attr_call, attr_var, loc)
    np_call = ir.Expr.call(attr_var, args, (), loc)
    arg_types = [typemap[x.name] for x in args]
    func_typ = func_var_typ.get_call_type(typingctx, arg_types, {})
    calltypes[np_call] = func_typ
    np_assign = ir.Assign(np_call, lhs, loc)
    return [g_np_assign, attr_assign, np_assign]