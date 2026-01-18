from __future__ import annotations
from sympy.core import Basic, S
from sympy.core.function import Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from functools import reduce
def print_glsl(expr, **settings):
    """Prints the GLSL representation of the given expression.

       See GLSLPrinter init function for settings.
    """
    print(glsl_code(expr, **settings))