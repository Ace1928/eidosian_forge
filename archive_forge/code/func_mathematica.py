from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable
import sympy
from sympy import Mul, Add, Pow, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning
def mathematica(s, additional_translations=None):
    sympy_deprecation_warning("The ``mathematica`` function for the Mathematica parser is now\ndeprecated. Use ``parse_mathematica`` instead.\nThe parameter ``additional_translation`` can be replaced by SymPy's\n.replace( ) or .subs( ) methods on the output expression instead.", deprecated_since_version='1.11', active_deprecations_target='mathematica-parser-new')
    parser = MathematicaParser(additional_translations)
    return sympify(parser._parse_old(s))