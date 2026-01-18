from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
from sympy.functions.elementary.trigonometric import (
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG
def fu(rv, measure=lambda x: (L(x), x.count_ops())):
    """Attempt to simplify expression by using transformation rules given
    in the algorithm by Fu et al.

    :func:`fu` will try to minimize the objective function ``measure``.
    By default this first minimizes the number of trig terms and then minimizes
    the number of total operations.

    Examples
    ========

    >>> from sympy.simplify.fu import fu
    >>> from sympy import cos, sin, tan, pi, S, sqrt
    >>> from sympy.abc import x, y, a, b

    >>> fu(sin(50)**2 + cos(50)**2 + sin(pi/6))
    3/2
    >>> fu(sqrt(6)*cos(x) + sqrt(2)*sin(x))
    2*sqrt(2)*sin(x + pi/3)

    CTR1 example

    >>> eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2
    >>> fu(eq)
    cos(x)**4 - 2*cos(y)**2 + 2

    CTR2 example

    >>> fu(S.Half - cos(2*x)/2)
    sin(x)**2

    CTR3 example

    >>> fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b)))
    sqrt(2)*sin(a + b + pi/4)

    CTR4 example

    >>> fu(sqrt(3)*cos(x)/2 + sin(x)/2)
    sin(x + pi/3)

    Example 1

    >>> fu(1-sin(2*x)**2/4-sin(y)**2-cos(x)**4)
    -cos(x)**2 + cos(y)**2

    Example 2

    >>> fu(cos(4*pi/9))
    sin(pi/18)
    >>> fu(cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9))
    1/16

    Example 3

    >>> fu(tan(7*pi/18)+tan(5*pi/18)-sqrt(3)*tan(5*pi/18)*tan(7*pi/18))
    -sqrt(3)

    Objective function example

    >>> fu(sin(x)/cos(x))  # default objective function
    tan(x)
    >>> fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) # maximize op count
    sin(x)/cos(x)

    References
    ==========

    .. [1] https://www.sciencedirect.com/science/article/pii/S0895717706001609
    """
    fRL1 = greedy(RL1, measure)
    fRL2 = greedy(RL2, measure)
    was = rv
    rv = sympify(rv)
    if not isinstance(rv, Expr):
        return rv.func(*[fu(a, measure=measure) for a in rv.args])
    rv = TR1(rv)
    if rv.has(tan, cot):
        rv1 = fRL1(rv)
        if measure(rv1) < measure(rv):
            rv = rv1
        if rv.has(tan, cot):
            rv = TR2(rv)
    if rv.has(sin, cos):
        rv1 = fRL2(rv)
        rv2 = TR8(TRmorrie(rv1))
        rv = min([was, rv, rv1, rv2], key=measure)
    return min(TR2i(rv), rv, key=measure)