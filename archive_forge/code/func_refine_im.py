from __future__ import annotations
from typing import Callable
from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
from sympy.core.logic import fuzzy_not
from sympy.logic.boolalg import Boolean
from sympy.assumptions import ask, Q  # type: ignore
def refine_im(expr, assumptions):
    """
    Handler for imaginary part.

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_im
    >>> from sympy import Q, im
    >>> from sympy.abc import x
    >>> refine_im(im(x), Q.real(x))
    0
    >>> refine_im(im(x), Q.imaginary(x))
    -I*x
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return S.Zero
    if ask(Q.imaginary(arg), assumptions):
        return -S.ImaginaryUnit * arg
    return _refine_reim(expr, assumptions)