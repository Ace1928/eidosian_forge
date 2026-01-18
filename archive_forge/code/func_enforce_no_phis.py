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
def enforce_no_phis(func_ir):
    """
    Enforce there being no ir.Expr.phi nodes in the IR.
    """
    for blk in func_ir.blocks.values():
        phis = [x for x in blk.find_exprs(op='phi')]
        if phis:
            msg = 'Illegal IR, phi found at: %s' % phis[0]
            raise CompilerError(msg, loc=phis[0].loc)