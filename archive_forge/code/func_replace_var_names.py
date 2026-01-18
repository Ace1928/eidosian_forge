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
def replace_var_names(blocks, namedict):
    """replace variables (ir.Var to ir.Var) from dictionary (name -> name)"""
    new_namedict = {}
    for l, r in namedict.items():
        if l != r:
            new_namedict[l] = r

    def replace_name(var, namedict):
        assert isinstance(var, ir.Var)
        while var.name in namedict:
            var = ir.Var(var.scope, namedict[var.name], var.loc)
        return var
    visit_vars(blocks, replace_name, new_namedict)