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
def remove_dead_block(block, lives, call_table, arg_aliases, alias_map, alias_set, func_ir, typemap):
    """remove dead code using liveness info.
    Mutable arguments (e.g. arrays) that are not definitely assigned are live
    after return of function.
    """
    removed = False
    new_body = [block.terminator]
    for stmt in reversed(block.body[:-1]):
        if config.DEBUG_ARRAY_OPT >= 2:
            print('remove_dead_block', stmt)
        alias_lives = set()
        init_alias_lives = lives & alias_set
        for v in init_alias_lives:
            alias_lives |= alias_map[v]
        lives_n_aliases = lives | alias_lives | arg_aliases
        if type(stmt) in remove_dead_extensions:
            f = remove_dead_extensions[type(stmt)]
            stmt = f(stmt, lives, lives_n_aliases, arg_aliases, alias_map, func_ir, typemap)
            if stmt is None:
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('Statement was removed.')
                removed = True
                continue
        if isinstance(stmt, ir.Assign):
            lhs = stmt.target
            rhs = stmt.value
            if lhs.name not in lives and has_no_side_effect(rhs, lives_n_aliases, call_table):
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('Statement was removed.')
                removed = True
                continue
            if isinstance(rhs, ir.Var) and lhs.name == rhs.name:
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('Statement was removed.')
                removed = True
                continue
        if isinstance(stmt, ir.Del):
            if stmt.value not in lives:
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('Statement was removed.')
                removed = True
                continue
        if isinstance(stmt, ir.SetItem):
            name = stmt.target.name
            if name not in lives_n_aliases:
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('Statement was removed.')
                continue
        if type(stmt) in analysis.ir_extension_usedefs:
            def_func = analysis.ir_extension_usedefs[type(stmt)]
            uses, defs = def_func(stmt)
            lives -= defs
            lives |= uses
        else:
            lives |= {v.name for v in stmt.list_vars()}
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr):
                    rhs_vars = {v.name for v in stmt.value.list_vars()}
                    if lhs.name not in rhs_vars:
                        lives.remove(lhs.name)
                else:
                    lives.remove(lhs.name)
        new_body.append(stmt)
    new_body.reverse()
    block.body = new_body
    return removed