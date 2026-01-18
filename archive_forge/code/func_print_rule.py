from __future__ import annotations
from typing import Any
from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
def print_rule(pos, val):
    """Helper function to print a rule of Mathematica"""
    return '{} -> {}'.format(self.doprint(pos), self.doprint(val))