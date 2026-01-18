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
def index_var_of_get_setitem(stmt):
    """get index variable for getitem/setitem nodes (and static cases)"""
    if is_getitem(stmt):
        if stmt.value.op == 'getitem':
            return stmt.value.index
        else:
            return stmt.value.index_var
    if is_setitem(stmt):
        if isinstance(stmt, ir.SetItem):
            return stmt.index
        else:
            return stmt.index_var
    return None