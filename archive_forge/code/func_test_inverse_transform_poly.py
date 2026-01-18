from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
def test_inverse_transform_poly():
    """
    This function tests the substitution x -> 1/x
    in rational functions represented using Poly.
    """
    fns = [(15 * x ** 3 - 8 * x ** 2 - 2 * x - 6) / (18 * x + 6), (180 * x ** 5 + 40 * x ** 4 + 80 * x ** 3 + 30 * x ** 2 - 60 * x - 80) / (180 * x ** 3 - 150 * x ** 2 + 75 * x + 12), (-15 * x ** 5 - 36 * x ** 4 + 75 * x ** 3 - 60 * x ** 2 - 80 * x - 60) / (80 * x ** 4 + 60 * x ** 3 + 60 * x ** 2 + 60 * x - 80), (60 * x ** 7 + 24 * x ** 6 - 15 * x ** 5 - 20 * x ** 4 + 30 * x ** 2 + 100 * x - 60) / (240 * x ** 2 - 20 * x - 30), (30 * x ** 6 - 12 * x ** 5 + 15 * x ** 4 - 15 * x ** 2 + 10 * x + 60) / (3 * x ** 10 - 45 * x ** 9 + 15 * x ** 5 + 15 * x ** 4 - 5 * x ** 3 + 15 * x ** 2 + 45 * x - 15)]
    for f in fns:
        num, den = [Poly(e, x) for e in f.as_numer_denom()]
        num, den = inverse_transform_poly(num, den, x)
        assert f.subs(x, 1 / x).cancel() == num / den