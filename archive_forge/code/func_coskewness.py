from sympy.sets import FiniteSet
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import FallingFactorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import piecewise_fold
from sympy.integrals.integrals import Integral
from sympy.solvers.solveset import solveset
from .rv import (probability, expectation, density, where, given, pspace, cdf, PSpace,
def coskewness(X, Y, Z, condition=None, **kwargs):
    """
    Calculates the co-skewness of three random variables.

    Explanation
    ===========

    Mathematically Coskewness is defined as

    .. math::
        coskewness(X,Y,Z)=\\frac{E[(X-E[X]) * (Y-E[Y]) * (Z-E[Z])]} {\\sigma_{X}\\sigma_{Y}\\sigma_{Z}}

    Parameters
    ==========

    X : RandomSymbol
            Random Variable used to calculate coskewness
    Y : RandomSymbol
            Random Variable used to calculate coskewness
    Z : RandomSymbol
            Random Variable used to calculate coskewness
    condition : Expr containing RandomSymbols
            A conditional expression

    Examples
    ========

    >>> from sympy.stats import coskewness, Exponential, skewness
    >>> from sympy import symbols
    >>> p = symbols('p', positive=True)
    >>> X = Exponential('X', p)
    >>> Y = Exponential('Y', 2*p)
    >>> coskewness(X, Y, Y)
    0
    >>> coskewness(X, Y + X, Y + 2*X)
    16*sqrt(85)/85
    >>> coskewness(X + 2*Y, Y + X, Y + 2*X, X > 3)
    9*sqrt(170)/85
    >>> coskewness(Y, Y, Y) == skewness(Y)
    True
    >>> coskewness(X, Y + p*X, Y + 2*p*X)
    4/(sqrt(1 + 1/(4*p**2))*sqrt(4 + 1/(4*p**2)))

    Returns
    =======

    coskewness : The coskewness of the three random variables

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coskewness

    """
    num = expectation((X - expectation(X, condition, **kwargs)) * (Y - expectation(Y, condition, **kwargs)) * (Z - expectation(Z, condition, **kwargs)), condition, **kwargs)
    den = std(X, condition, **kwargs) * std(Y, condition, **kwargs) * std(Z, condition, **kwargs)
    return num / den