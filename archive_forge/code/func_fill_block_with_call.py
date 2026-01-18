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
def fill_block_with_call(newblock, callee, label_next, inputs, outputs):
    """Fill *newblock* to call *callee* with arguments listed in *inputs*.
    The returned values are unwrapped into variables in *outputs*.
    The block would then jump to *label_next*.
    """
    scope = newblock.scope
    loc = newblock.loc
    fn = ir.Const(value=callee, loc=loc)
    fnvar = scope.make_temp(loc=loc)
    newblock.append(ir.Assign(target=fnvar, value=fn, loc=loc))
    args = [scope.get_exact(name) for name in inputs]
    callexpr = ir.Expr.call(func=fnvar, args=args, kws=(), loc=loc)
    callres = scope.make_temp(loc=loc)
    newblock.append(ir.Assign(target=callres, value=callexpr, loc=loc))
    for i, out in enumerate(outputs):
        target = scope.get_exact(out)
        getitem = ir.Expr.static_getitem(value=callres, index=i, index_var=None, loc=loc)
        newblock.append(ir.Assign(target=target, value=getitem, loc=loc))
    newblock.append(ir.Jump(target=label_next, loc=loc))
    return newblock