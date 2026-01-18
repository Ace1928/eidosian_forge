from __future__ import annotations
from typing import Any
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def print_jscode(expr, **settings):
    """Prints the Javascript representation of the given expression.

       See jscode for the meaning of the optional arguments.
    """
    print(jscode(expr, **settings))