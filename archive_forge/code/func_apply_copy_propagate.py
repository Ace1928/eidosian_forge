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
def apply_copy_propagate(blocks, in_copies, name_var_table, typemap, calltypes, save_copies=None):
    """apply copy propagation to IR: replace variables when copies available"""
    if save_copies is None:
        save_copies = []
    for label, block in blocks.items():
        var_dict = {l: name_var_table[r] for l, r in in_copies[label]}
        for stmt in block.body:
            if type(stmt) in apply_copy_propagate_extensions:
                f = apply_copy_propagate_extensions[type(stmt)]
                f(stmt, var_dict, name_var_table, typemap, calltypes, save_copies)
            elif isinstance(stmt, ir.Assign):
                stmt.value = replace_vars_inner(stmt.value, var_dict)
            else:
                replace_vars_stmt(stmt, var_dict)
            fix_setitem_type(stmt, typemap, calltypes)
            for T, f in copy_propagate_extensions.items():
                if isinstance(stmt, T):
                    gen_set, kill_set = f(stmt, typemap)
                    for lhs, rhs in gen_set:
                        if rhs in name_var_table:
                            var_dict[lhs] = name_var_table[rhs]
                    for l, r in var_dict.copy().items():
                        if l in kill_set or r.name in kill_set:
                            var_dict.pop(l)
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                lhs = stmt.target.name
                rhs = stmt.value.name
                if lhs != rhs:
                    if typemap[lhs] == typemap[rhs] and rhs in name_var_table:
                        var_dict[lhs] = name_var_table[rhs]
                    else:
                        var_dict.pop(lhs, None)
                    lhs_kill = []
                    for k, v in var_dict.items():
                        if v.name == lhs:
                            lhs_kill.append(k)
                    for k in lhs_kill:
                        var_dict.pop(k, None)
            if isinstance(stmt, ir.Assign) and (not isinstance(stmt.value, ir.Var)):
                lhs = stmt.target.name
                var_dict.pop(lhs, None)
                lhs_kill = []
                for k, v in var_dict.items():
                    if v.name == lhs:
                        lhs_kill.append(k)
                for k in lhs_kill:
                    var_dict.pop(k, None)
        save_copies.extend(var_dict.items())
    return save_copies