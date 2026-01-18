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
def replace_vars(blocks, vardict):
    """replace variables (ir.Var to ir.Var) from dictionary (name -> ir.Var)"""
    new_vardict = {}
    for l, r in vardict.items():
        if l != r.name:
            new_vardict[l] = r
    visit_vars(blocks, replace_var_callback, new_vardict)