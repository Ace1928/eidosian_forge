import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def rvsdg_to_ir(func_id: bytecode.FunctionIdentity, rvsdg: SCFG) -> ir.FunctionIR:
    rvsdg2ir = RVSDG2IR(func_id)
    data = rvsdg2ir.initialize()
    rvsdg2ir.visit_graph(rvsdg, data)
    rvsdg2ir.finalize()
    for blk in rvsdg2ir.blocks.values():
        blk.verify()
    defs = ir_utils.build_definitions(rvsdg2ir.blocks)
    fir = ir.FunctionIR(blocks=rvsdg2ir.blocks, is_generator=False, func_id=func_id, loc=rvsdg2ir.first_loc, definitions=defs, arg_count=len(func_id.arg_names), arg_names=func_id.arg_names)
    if DEBUG_GRAPH:
        fir.render_dot().view()
    return fir