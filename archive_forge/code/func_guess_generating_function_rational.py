from sympy.concrete.products import (Product, product)
from sympy.core import Function, S
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import integrate
from sympy.polys.polyfuncs import rational_interpolate as rinterp
from sympy.polys.polytools import lcm
from sympy.simplify.radsimp import denom
from sympy.utilities import public
@public
def guess_generating_function_rational(v, X=Symbol('x')):
    """
    Tries to "guess" a rational generating function for a sequence of rational
    numbers v.

    Examples
    ========

    >>> from sympy.concrete.guess import guess_generating_function_rational
    >>> from sympy import fibonacci
    >>> l = [fibonacci(k) for k in range(5,15)]
    >>> guess_generating_function_rational(l)
    (3*x + 5)/(-x**2 - x + 1)

    See Also
    ========

    sympy.series.approximants
    mpmath.pade

    """
    q = find_simple_recurrence_vector(v)
    n = len(q)
    if n <= 1:
        return None
    p = [sum((v[i - k] * q[k] for k in range(min(i + 1, n)))) for i in range(len(v) >> 1)]
    return sum((p[k] * X ** k for k in range(len(p)))) / sum((q[k] * X ** k for k in range(n)))