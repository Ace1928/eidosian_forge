from collections import Counter
from sympy.core import Mul, sympify
from sympy.core.add import Add
from sympy.core.expr import ExprBuilder
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import log
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix
from sympy.strategies import (
from sympy.utilities.exceptions import sympy_deprecation_warning
def hadamard_power(base, exp):
    base = sympify(base)
    exp = sympify(exp)
    if exp == 1:
        return base
    if not base.is_Matrix:
        return base ** exp
    if exp.is_Matrix:
        raise ValueError('cannot raise expression to a matrix')
    return HadamardPower(base, exp)