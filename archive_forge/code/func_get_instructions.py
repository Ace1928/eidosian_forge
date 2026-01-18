import sys
import types
import collections
import io
from opcode import *
from opcode import (
def get_instructions(x, *, first_line=None, show_caches=False, adaptive=False):
    """Iterator for the opcodes in methods, functions or code

    Generates a series of Instruction named tuples giving the details of
    each operations in the supplied code.

    If *first_line* is not None, it indicates the line number that should
    be reported for the first source line in the disassembled code.
    Otherwise, the source line information (if any) is taken directly from
    the disassembled code object.
    """
    co = _get_code_object(x)
    linestarts = dict(findlinestarts(co))
    if first_line is not None:
        line_offset = first_line - co.co_firstlineno
    else:
        line_offset = 0
    return _get_instructions_bytes(_get_code_array(co, adaptive), co._varname_from_oparg, co.co_names, co.co_consts, linestarts, line_offset, co_positions=co.co_positions(), show_caches=show_caches)