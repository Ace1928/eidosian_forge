import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def build_line_to_contents(self):
    line_to_contents = {}
    instructions = self.instructions
    while instructions:
        s = self._next_instruction_to_str(line_to_contents)
        if s is RESTART_FROM_LOOKAHEAD:
            continue
        if s is None:
            continue
        _MsgPart.add_to_line_to_contents(s, line_to_contents)
        m = self.max_line(s)
        if m != self.SMALL_LINE_INT:
            line_to_contents.setdefault(m, []).append(SEPARATOR)
    return line_to_contents