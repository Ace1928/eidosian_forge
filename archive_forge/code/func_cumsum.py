from __future__ import annotations
from ..runtime.jit import jit
from . import core, math
@jit
@core._add_scan_docstr('cumsum')
def cumsum(input, axis=0):
    input = core._promote_reduction_input(input)
    return core.associative_scan(input, axis, _sum_combine)