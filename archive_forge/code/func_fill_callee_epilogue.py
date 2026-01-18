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
def fill_callee_epilogue(block, outputs):
    """
    Fill a new block *block* to prepare the return values.
    This block is the last block of the function.

    Expected to use with *fill_block_with_call()*
    """
    scope = block.scope
    loc = block.loc
    vals = [scope.get_exact(name=name) for name in outputs]
    tupexpr = ir.Expr.build_tuple(items=vals, loc=loc)
    tup = scope.make_temp(loc=loc)
    block.append(ir.Assign(target=tup, value=tupexpr, loc=loc))
    block.append(ir.Return(value=tup, loc=loc))
    return block