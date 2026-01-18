from __future__ import annotations
from typing import Any
from functools import wraps
from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def print_fcode(expr, **settings):
    """Prints the Fortran representation of the given expression.

       See fcode for the meaning of the optional arguments.
    """
    print(fcode(expr, **settings))