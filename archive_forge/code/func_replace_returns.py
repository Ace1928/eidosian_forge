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
def replace_returns(blocks, target, return_label):
    """
    Return return statement by assigning directly to target, and a jump.
    """
    for block in blocks.values():
        if not block.body:
            continue
        stmt = block.terminator
        if isinstance(stmt, ir.Return):
            block.body.pop()
            cast_stmt = block.body.pop()
            assert isinstance(cast_stmt, ir.Assign) and isinstance(cast_stmt.value, ir.Expr) and (cast_stmt.value.op == 'cast'), 'invalid return cast'
            block.body.append(ir.Assign(cast_stmt.value.value, target, stmt.loc))
            block.body.append(ir.Jump(return_label, stmt.loc))