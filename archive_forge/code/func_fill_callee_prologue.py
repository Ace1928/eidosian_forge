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
def fill_callee_prologue(block, inputs, label_next):
    """
    Fill a new block *block* that unwraps arguments using names in *inputs* and
    then jumps to *label_next*.

    Expected to use with *fill_block_with_call()*
    """
    scope = block.scope
    loc = block.loc
    args = [ir.Arg(name=k, index=i, loc=loc) for i, k in enumerate(inputs)]
    for aname, aval in zip(inputs, args):
        tmp = ir.Var(scope=scope, name=aname, loc=loc)
        block.append(ir.Assign(target=tmp, value=aval, loc=loc))
    block.append(ir.Jump(target=label_next, loc=loc))
    return block