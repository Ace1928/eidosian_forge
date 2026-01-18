from __future__ import annotations
from typing import Any
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
def print_rcode(expr, **settings):
    """Prints R representation of the given expression."""
    print(rcode(expr, **settings))