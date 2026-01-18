from __future__ import annotations
from typing import Any
from collections import defaultdict
from itertools import chain
import string
from sympy.codegen.ast import (
from sympy.codegen.fnodes import (
from sympy.core import S, Add, N, Float, Symbol
from sympy.core.function import Function
from sympy.core.numbers import equal_valued
from sympy.core.relational import Eq
from sympy.sets import Range
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.printing.printer import printer_context
from sympy.printing.codeprinter import fcode, print_fcode # noqa:F401
def split_pos_code(line, endpos):
    if len(line) <= endpos:
        return len(line)
    pos = endpos
    split = lambda pos: line[pos] in my_alnum and line[pos - 1] not in my_alnum or (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or (line[pos] in my_white and line[pos - 1] not in my_white) or (line[pos] not in my_white and line[pos - 1] in my_white)
    while not split(pos):
        pos -= 1
        if pos == 0:
            return endpos
    return pos