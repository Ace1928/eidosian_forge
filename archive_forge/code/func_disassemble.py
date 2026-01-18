import sys
import types
import collections
import io
from opcode import *
from opcode import (
def disassemble(co, lasti=-1, *, file=None, show_caches=False, adaptive=False):
    """Disassemble a code object."""
    linestarts = dict(findlinestarts(co))
    exception_entries = _parse_exception_table(co)
    _disassemble_bytes(_get_code_array(co, adaptive), lasti, co._varname_from_oparg, co.co_names, co.co_consts, linestarts, file=file, exception_entries=exception_entries, co_positions=co.co_positions(), show_caches=show_caches)