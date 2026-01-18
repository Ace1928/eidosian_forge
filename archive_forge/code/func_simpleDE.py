from collections import defaultdict
from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable
def simpleDE(f, x, g, order=4):
    """
    Generates simple DE.

    Explanation
    ===========

    DE is of the form

    .. math::
        f^k(x) + \\sum\\limits_{j=0}^{k-1} A_j f^j(x) = 0

    where :math:`A_j` should be rational function in x.

    Generates DE's upto order 4 (default). DE's can also have free parameters.

    By increasing order, higher order DE's can be found.

    Yields a tuple of (DE, order).
    """
    from sympy.solvers.solveset import linsolve
    a = symbols('a:%d' % order)

    def _makeDE(k):
        eq = f.diff(x, k) + Add(*[a[i] * f.diff(x, i) for i in range(0, k)])
        DE = g(x).diff(x, k) + Add(*[a[i] * g(x).diff(x, i) for i in range(0, k)])
        return (eq, DE)
    found = False
    for k in range(1, order + 1):
        eq, DE = _makeDE(k)
        eq = eq.expand()
        terms = eq.as_ordered_terms()
        ind = rational_independent(terms, x)
        if found or len(ind) == k:
            sol = dict(zip(a, (i for s in linsolve(ind, a[:k]) for i in s)))
            if sol:
                found = True
                DE = DE.subs(sol)
            DE = DE.as_numer_denom()[0]
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            yield (DE.collect(Derivative(g(x))), k)