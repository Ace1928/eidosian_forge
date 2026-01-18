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
def is_operator_or_getitem(expr):
    """true if expr is unary or binary operator or getitem"""
    return isinstance(expr, ir.Expr) and getattr(expr, 'op', False) and (expr.op in ['unary', 'binop', 'inplace_binop', 'getitem', 'static_getitem'])