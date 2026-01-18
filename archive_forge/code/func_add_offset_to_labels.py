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
def add_offset_to_labels(blocks, offset):
    """add an offset to all block labels and jump/branch targets
    """
    new_blocks = {}
    for l, b in blocks.items():
        term = None
        if b.body:
            term = b.body[-1]
            for inst in b.body:
                for T, f in add_offset_to_labels_extensions.items():
                    if isinstance(inst, T):
                        f_max = f(inst, offset)
        if isinstance(term, ir.Jump):
            b.body[-1] = ir.Jump(term.target + offset, term.loc)
        if isinstance(term, ir.Branch):
            b.body[-1] = ir.Branch(term.cond, term.truebr + offset, term.falsebr + offset, term.loc)
        new_blocks[l + offset] = b
    return new_blocks