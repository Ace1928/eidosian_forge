from __future__ import annotations
from typing import Any
from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search
def print_octave_code(expr, **settings):
    """Prints the Octave (or Matlab) representation of the given expression.

    See `octave_code` for the meaning of the optional arguments.
    """
    print(octave_code(expr, **settings))