from sympy.core import S, sympify
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise, piecewise_fold
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from functools import lru_cache
def bspline_basis_set(d, knots, x):
    """
    Return the ``len(knots)-d-1`` B-splines at *x* of degree *d*
    with *knots*.

    Explanation
    ===========

    This function returns a list of piecewise polynomials that are the
    ``len(knots)-d-1`` B-splines of degree *d* for the given knots.
    This function calls ``bspline_basis(d, knots, n, x)`` for different
    values of *n*.

    Examples
    ========

    >>> from sympy import bspline_basis_set
    >>> from sympy.abc import x
    >>> d = 2
    >>> knots = range(5)
    >>> splines = bspline_basis_set(d, knots, x)
    >>> splines
    [Piecewise((x**2/2, (x >= 0) & (x <= 1)),
               (-x**2 + 3*x - 3/2, (x >= 1) & (x <= 2)),
               (x**2/2 - 3*x + 9/2, (x >= 2) & (x <= 3)),
               (0, True)),
    Piecewise((x**2/2 - x + 1/2, (x >= 1) & (x <= 2)),
              (-x**2 + 5*x - 11/2, (x >= 2) & (x <= 3)),
              (x**2/2 - 4*x + 8, (x >= 3) & (x <= 4)),
              (0, True))]

    Parameters
    ==========

    d : integer
        degree of bspline

    knots : list of integers
        list of knots points of bspline

    x : symbol

    See Also
    ========

    bspline_basis

    """
    n_splines = len(knots) - d - 1
    return [bspline_basis(d, tuple(knots), i, x) for i in range(n_splines)]