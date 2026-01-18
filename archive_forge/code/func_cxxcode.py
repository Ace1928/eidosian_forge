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
def cxxcode(expr, assign_to=None, standard='c++11', **settings):
    """ C++ equivalent of :func:`~.ccode`. """
    from sympy.printing.cxx import cxx_code_printers
    return cxx_code_printers[standard.lower()](settings).doprint(expr, assign_to)