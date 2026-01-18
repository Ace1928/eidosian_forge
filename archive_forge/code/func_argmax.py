from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
@core._add_reduction_docstr('maximum index', tie_break_arg='tie_break_left')
def argmax(input, axis, tie_break_left=True):
    _, ret = max(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left)
    return ret