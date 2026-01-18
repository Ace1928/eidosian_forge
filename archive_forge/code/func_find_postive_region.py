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
def find_postive_region():
    found = False
    for idx in reversed(range(len(blk.body))):
        stmt = blk.body[idx]
        if isinstance(stmt, ir.Assign):
            value = stmt.value
            if isinstance(value, ir.Expr) and value.op == 'list_to_tuple':
                target_list = value.info[0]
                found = True
                bt = (idx, stmt)
        if found:
            if isinstance(stmt, ir.Assign):
                if stmt.target.name == target_list:
                    region = (bt, (idx, stmt))
                    return region