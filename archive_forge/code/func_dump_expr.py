from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def dump_expr(self, expr, cu_offset=None):
    """ Parse and dump a DWARF expression. expr should be a list of
            (integer) byte values. cu_offset is the cu_offset
            value from the CU object where the expression resides.
            Only affects a handful of GNU opcodes, if None is provided,
            that's not a crash condition, only the expression dump will
            not be consistent of that of readelf.

            Returns a string representing the expression.
        """
    parsed = self.expr_parser.parse_expr(expr)
    s = []
    for deo in parsed:
        s.append(self._dump_to_string(deo.op, deo.op_name, deo.args, cu_offset))
    return '; '.join(s)