from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.numbers import igcdex, Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit
@cacheit
def cos_17() -> Expr:
    """Computes $\\cos \\frac{\\pi}{17}$ in square roots"""
    return sqrt((15 + sqrt(17)) / 32 + sqrt(2) * (sqrt(17 - sqrt(17)) + sqrt(sqrt(2) * (-8 * sqrt(17 + sqrt(17)) - (1 - sqrt(17)) * sqrt(17 - sqrt(17))) + 6 * sqrt(17) + 34)) / 32)