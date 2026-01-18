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
def get_array_accesses(blocks, accesses=None):
    """returns a set of arrays accessed and their indices.
    """
    if accesses is None:
        accesses = set()
    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.SetItem):
                accesses.add((inst.target.name, inst.index.name))
            if isinstance(inst, ir.StaticSetItem):
                accesses.add((inst.target.name, inst.index_var.name))
            if isinstance(inst, ir.Assign):
                lhs = inst.target.name
                rhs = inst.value
                if isinstance(rhs, ir.Expr) and rhs.op == 'getitem':
                    accesses.add((rhs.value.name, rhs.index.name))
                if isinstance(rhs, ir.Expr) and rhs.op == 'static_getitem':
                    index = rhs.index
                    if index is None or is_slice_index(index):
                        index = rhs.index_var.name
                    accesses.add((rhs.value.name, index))
            for T, f in array_accesses_extensions.items():
                if isinstance(inst, T):
                    f(inst, accesses)
    return accesses