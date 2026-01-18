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
def interpret_bytecode(self, op: Op):
    """Interpret a single Op containing bytecode instructions.

        Internally, it dispatches to methods with names following the pattern
        `op_<opname>`.
        """
    assert op.bc_inst is not None
    pos = op.bc_inst.positions
    assert pos is not None
    self.loc = self.loc.with_lineno(pos.lineno, pos.col_offset)
    if self._emit_debug_print:
        where = f'{op.bc_inst.offset:3}:({pos.lineno:3}:{pos.col_offset:3})'
        msg = f'[{where}] {op.bc_inst.opname}({op.bc_inst.argrepr}) '
        self.debug_print(msg)
        for k, vs in op.input_ports.items():
            val = self.vsmap.get(vs, None)
            if val is None:
                self.debug_print(f'   in {k:>6}: <undef>')
            else:
                self.debug_print(f'   in {k:>6}:', val)
    fn = getattr(self, f'op_{op.bc_inst.opname}')
    fn(op, op.bc_inst)
    if self._emit_debug_print:
        for k, vs in op.output_ports.items():
            val = self.vsmap.get(vs, None)
            if val is None:
                self.debug_print(f'  out {k:>6}: <undef>')
            else:
                self.debug_print(f'  out {k:>6}:', val)