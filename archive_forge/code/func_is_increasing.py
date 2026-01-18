from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import sec, csc, cot, tan, cos
from sympy.utilities.misc import filldedent
def is_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is increasing (either strictly increasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_increasing(-x**2, Interval(-oo, 0))
    True
    >>> is_increasing(-x**2, Interval(0, oo))
    False
    >>> is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    False
    >>> is_increasing(x**2 + y, Interval(1, 2), x)
    True

    """
    return monotonicity_helper(expression, lambda x: x >= 0, interval, symbol)