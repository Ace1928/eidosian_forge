from __future__ import annotations
from typing import Any
from functools import wraps
from itertools import chain
from sympy.core import S
from sympy.core.numbers import equal_valued
from sympy.codegen.ast import (
from sympy.printing.codeprinter import CodePrinter, requires
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
from sympy.printing.codeprinter import ccode, print_ccode # noqa:F401
def get_math_macros():
    """ Returns a dictionary with math-related macros from math.h/cmath

    Note that these macros are not strictly required by the C/C++-standard.
    For MSVC they are enabled by defining "_USE_MATH_DEFINES" (preferably
    via a compilation flag).

    Returns
    =======

    Dictionary mapping SymPy expressions to strings (macro names)

    """
    from sympy.codegen.cfunctions import log2, Sqrt
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.miscellaneous import sqrt
    return {S.Exp1: 'M_E', log2(S.Exp1): 'M_LOG2E', 1 / log(2): 'M_LOG2E', log(2): 'M_LN2', log(10): 'M_LN10', S.Pi: 'M_PI', S.Pi / 2: 'M_PI_2', S.Pi / 4: 'M_PI_4', 1 / S.Pi: 'M_1_PI', 2 / S.Pi: 'M_2_PI', 2 / sqrt(S.Pi): 'M_2_SQRTPI', 2 / Sqrt(S.Pi): 'M_2_SQRTPI', sqrt(2): 'M_SQRT2', Sqrt(2): 'M_SQRT2', 1 / sqrt(2): 'M_SQRT1_2', 1 / Sqrt(2): 'M_SQRT1_2'}