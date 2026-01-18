from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
from collections import defaultdict
from typing import Tuple as tTuple
def decompose_power_rat(expr: Expr) -> tTuple[Expr, Rational]:
    """
    Decompose power into symbolic base and rational exponent;
    if the exponent is not a Rational, then separate only the
    integer coefficient.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power_rat
    >>> from sympy.abc import x
    >>> from sympy import sqrt, exp

    >>> decompose_power_rat(sqrt(x))
    (x, 1/2)
    >>> decompose_power_rat(exp(-3*x/2))
    (exp(x/2), -3)

    """
    _ = base, exp = expr.as_base_exp()
    return _ if exp.is_Rational else decompose_power(expr)