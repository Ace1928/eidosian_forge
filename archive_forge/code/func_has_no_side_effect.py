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
def has_no_side_effect(rhs, lives, call_table):
    """ Returns True if this expression has no side effects that
        would prevent re-ordering.
    """
    from numba.parfors import array_analysis, parfor
    from numba.misc.special import prange
    if isinstance(rhs, ir.Expr) and rhs.op == 'call':
        func_name = rhs.func.name
        if func_name not in call_table or call_table[func_name] == []:
            return False
        call_list = call_table[func_name]
        if call_list == ['empty', numpy] or call_list == [slice] or call_list == ['stencil', numba] or (call_list == ['log', numpy]) or (call_list == ['dtype', numpy]) or (call_list == [array_analysis.wrap_index]) or (call_list == [prange]) or (call_list == ['prange', numba]) or (call_list == ['pndindex', numba]) or (call_list == [parfor.internal_prange]) or (call_list == ['ceil', math]) or (call_list == [max]) or (call_list == [int]):
            return True
        elif isinstance(call_list[0], _Intrinsic) and (call_list[0]._name == 'empty_inferred' or call_list[0]._name == 'unsafe_empty_inferred'):
            return True
        from numba.core.registry import CPUDispatcher
        from numba.np.linalg import dot_3_mv_check_args
        if isinstance(call_list[0], CPUDispatcher):
            py_func = call_list[0].py_func
            if py_func == dot_3_mv_check_args:
                return True
        for f in remove_call_handlers:
            if f(rhs, lives, call_list):
                return True
        return False
    if isinstance(rhs, ir.Expr) and rhs.op == 'inplace_binop':
        return rhs.lhs.name not in lives
    if isinstance(rhs, ir.Yield):
        return False
    if isinstance(rhs, ir.Expr) and rhs.op == 'pair_first':
        return False
    return True