from __future__ import annotations
from typing import Any
from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
def mathematica_code(expr, **settings):
    """Converts an expr to a string of the Wolfram Mathematica code

    Examples
    ========

    >>> from sympy import mathematica_code as mcode, symbols, sin
    >>> x = symbols('x')
    >>> mcode(sin(x).series(x).removeO())
    '(1/120)*x^5 - 1/6*x^3 + x'
    """
    return MCodePrinter(settings).doprint(expr)