from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Mul, Pow
from sympy.core.logic import fuzzy_not, fuzzy_and, fuzzy_or
from sympy.core.numbers import E, ImaginaryUnit, NaN, I, pi
from sympy.functions import Abs, acos, acot, asin, atan, exp, factorial, log
from sympy.matrices import Determinant, Trace
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.multipledispatch import MDNotImplementedError
from ..predicates.order import (NegativePredicate, NonNegativePredicate,
def nonz(i):
    if i._prec != 1:
        return i != 0