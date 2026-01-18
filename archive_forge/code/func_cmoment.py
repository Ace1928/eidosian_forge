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
def cmoment(X, n, condition=None, *, evaluate=True, **kwargs):
    """
    Return the nth central moment of a random expression about its mean.

    .. math::
        cmoment(X, n) = E((X - E(X))^{n})

    Examples
    ========

    >>> from sympy.stats import Die, cmoment, variance
    >>> X = Die('X', 6)
    >>> cmoment(X, 3)
    0
    >>> cmoment(X, 2)
    35/12
    >>> cmoment(X, 2) == variance(X)
    True
    """
    from sympy.stats.symbolic_probability import CentralMoment
    if evaluate:
        return CentralMoment(X, n, condition).doit()
    return CentralMoment(X, n, condition).rewrite(Integral)