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
def find_outer_value(func_ir, var):
    """Check if a variable is a global value, and return the value,
    or raise GuardException otherwise.
    """
    dfn = get_definition(func_ir, var)
    if isinstance(dfn, (ir.Global, ir.FreeVar)):
        return dfn.value
    if isinstance(dfn, ir.Expr) and dfn.op == 'getattr':
        prev_val = find_outer_value(func_ir, dfn.value)
        try:
            val = getattr(prev_val, dfn.attr)
            return val
        except AttributeError:
            raise GuardException
    raise GuardException