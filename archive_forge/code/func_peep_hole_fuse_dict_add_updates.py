import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def peep_hole_fuse_dict_add_updates(func_ir):
    """
    This rewrite removes d1._update_from_bytecode(d2)
    calls that are between two dictionaries, d1 and d2,
    in the same basic block. This pattern can appear as a
    result of Python 3.10 bytecode emission changes, which
    prevent large constant literal dictionaries
    (> 15 elements) from being constant. If both dictionaries
    are constant dictionaries defined in the same block and
    neither is used between the update call, then we replace d1
    with a new definition that combines the two dictionaries. At
    the bytecode translation stage we convert DICT_UPDATE into
    _update_from_bytecode, so we know that _update_from_bytecode
    always comes from the bytecode change and not user code.

    Python 3.10 may also rewrite the individual dictionaries
    as an empty build_map + many map_add. Here we again look
    for an _update_from_bytecode, and if so we replace these
    with a single constant dictionary.

    When running this algorithm we can always safely remove d2.

    This is the relevant section of the CPython 3.10 that causes
    this bytecode change:
    https://github.com/python/cpython/blob/3.10/Python/compile.c#L4048
    """
    errmsg = textwrap.dedent('\n        A DICT_UPDATE op-code was encountered that could not be replaced.\n        If you have created a large constant dictionary, this may\n        be an an indication that you are using inlined control\n        flow. You can resolve this issue by moving the control flow out of\n        the dicitonary constructor. For example, if you have\n\n            d = {a: 1 if flag else 0, ...)\n\n        Replace that with:\n\n            a_val = 1 if flag else 0\n            d = {a: a_val, ...)')
    already_deleted_defs = collections.defaultdict(set)
    for blk in func_ir.blocks.values():
        new_body = []
        lit_map_def_idx = {}
        lit_map_use_idx = collections.defaultdict(list)
        map_updates = {}
        blk_changed = False
        for i, stmt in enumerate(blk.body):
            new_inst = stmt
            stmt_build_map_out = None
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                if stmt.value.op == 'build_map':
                    stmt_build_map_out = stmt.target.name
                    lit_map_def_idx[stmt.target.name] = i
                    lit_map_use_idx[stmt.target.name].append(i)
                    map_updates[stmt.target.name] = stmt.value.items.copy()
                elif stmt.value.op == 'call' and i > 0:
                    func_name = stmt.value.func.name
                    getattr_stmt = blk.body[i - 1]
                    args = stmt.value.args
                    if isinstance(getattr_stmt, ir.Assign) and getattr_stmt.target.name == func_name and isinstance(getattr_stmt.value, ir.Expr) and (getattr_stmt.value.op == 'getattr') and (getattr_stmt.value.attr in ('__setitem__', '_update_from_bytecode')):
                        update_map_name = getattr_stmt.value.value.name
                        attr = getattr_stmt.value.attr
                        if attr == '__setitem__' and update_map_name in lit_map_use_idx:
                            map_updates[update_map_name].append(args)
                            lit_map_use_idx[update_map_name].extend([i - 1, i])
                        elif attr == '_update_from_bytecode':
                            d2_map_name = args[0].name
                            if update_map_name in lit_map_use_idx and d2_map_name in lit_map_use_idx:
                                map_updates[update_map_name].extend(map_updates[d2_map_name])
                                lit_map_use_idx[update_map_name].extend(lit_map_use_idx[d2_map_name])
                                lit_map_use_idx[update_map_name].append(i - 1)
                                for linenum in lit_map_use_idx[update_map_name]:
                                    _remove_assignment_definition(blk.body, linenum, func_ir, already_deleted_defs)
                                    new_body[linenum] = None
                                del lit_map_def_idx[d2_map_name]
                                del lit_map_use_idx[d2_map_name]
                                del map_updates[d2_map_name]
                                _remove_assignment_definition(blk.body, i, func_ir, already_deleted_defs)
                                new_inst = _build_new_build_map(func_ir, update_map_name, blk.body, lit_map_def_idx[update_map_name], map_updates[update_map_name])
                                lit_map_use_idx[update_map_name].clear()
                                lit_map_use_idx[update_map_name].append(i)
                                blk_changed = True
                            else:
                                raise UnsupportedError(errmsg)
            if not (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'getattr') and (stmt.value.value.name in lit_map_use_idx) and (stmt.value.attr in ('__setitem__', '_update_from_bytecode'))):
                for var in stmt.list_vars():
                    if var.name in lit_map_use_idx and var.name != stmt_build_map_out:
                        del lit_map_def_idx[var.name]
                        del lit_map_use_idx[var.name]
                        del map_updates[var.name]
            new_body.append(new_inst)
        if blk_changed:
            blk.body.clear()
            blk.body.extend([x for x in new_body if x is not None])
    return func_ir