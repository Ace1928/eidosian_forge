from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
@core._add_reduction_docstr('sum')
def sum(input, axis=None):
    input = core._promote_reduction_input(input)
    return core.reduce(input, axis, _sum_combine)