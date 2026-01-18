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
def peep_hole_delete_with_exit(func_ir):
    """
    This rewrite removes variables used to store the `__exit__` function
    loaded by SETUP_WITH.
    """
    dead_vars = set()
    for blk in func_ir.blocks.values():
        for stmt in blk.body:
            used = set(stmt.list_vars())
            for v in used:
                if v.name.startswith('$setup_with_exitfn'):
                    dead_vars.add(v)
            if used & dead_vars:
                if isinstance(stmt, ir.Assign):
                    dead_vars.add(stmt.target)
        new_body = []
        for stmt in blk.body:
            if not set(stmt.list_vars()) & dead_vars:
                new_body.append(stmt)
        blk.body.clear()
        blk.body.extend(new_body)
    return func_ir