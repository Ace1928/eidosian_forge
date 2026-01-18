from typing import Tuple as tTuple
from sympy.calculus.singularities import is_decreasing
from sympy.calculus.accumulationbounds import AccumulationBounds
from .expr_with_intlimits import ExprWithIntLimits
from .expr_with_limits import AddWithLimits
from .gosper import gosper_sum
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Float, _illegal
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Wild, Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cot, csc
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError, PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr, Poly, factor
from sympy.polys.rationaltools import together
from sympy.series.limitseq import limit_seq
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.sets.sets import FiniteSet, Interval
from sympy.utilities.iterables import sift
import itertools
def summation(f, *symbols, **kwargs):
    """
    Compute the summation of f with respect to symbols.

    Explanation
    ===========

    The notation for symbols is similar to the notation used in Integral.
    summation(f, (i, a, b)) computes the sum of f with respect to i from a to b,
    i.e.,

    ::

                                    b
                                  ____
                                  \\   `
        summation(f, (i, a, b)) =  )    f
                                  /___,
                                  i = a

    If it cannot compute the sum, it returns an unevaluated Sum object.
    Repeated sums can be computed by introducing additional symbols tuples::

    Examples
    ========

    >>> from sympy import summation, oo, symbols, log
    >>> i, n, m = symbols('i n m', integer=True)

    >>> summation(2*i - 1, (i, 1, n))
    n**2
    >>> summation(1/2**i, (i, 0, oo))
    2
    >>> summation(1/log(n)**n, (n, 2, oo))
    Sum(log(n)**(-n), (n, 2, oo))
    >>> summation(i, (i, 0, n), (n, 0, m))
    m**3/6 + m**2/2 + m/3

    >>> from sympy.abc import x
    >>> from sympy import factorial
    >>> summation(x**n/factorial(n), (n, 0, oo))
    exp(x)

    See Also
    ========

    Sum
    Product, sympy.concrete.products.product

    """
    return Sum(f, *symbols, **kwargs).doit(deep=False)