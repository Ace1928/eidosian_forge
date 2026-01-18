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
def fix_setitem_type(stmt, typemap, calltypes):
    """Copy propagation can replace setitem target variable, which can be array
    with 'A' layout. The replaced variable can be 'C' or 'F', so we update
    setitem call type reflect this (from matrix power test)
    """
    if not isinstance(stmt, (ir.SetItem, ir.StaticSetItem)):
        return
    t_typ = typemap[stmt.target.name]
    s_typ = calltypes[stmt].args[0]
    if not isinstance(s_typ, types.npytypes.Array) or not isinstance(t_typ, types.npytypes.Array):
        return
    if s_typ.layout == 'A' and t_typ.layout != 'A':
        new_s_typ = s_typ.copy(layout=t_typ.layout)
        calltypes[stmt].args = (new_s_typ, calltypes[stmt].args[1], calltypes[stmt].args[2])
    return