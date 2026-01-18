from sympy.concrete.products import Product
from sympy.core.function import expand_func
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (ff, rf, binomial, factorial, factorial2)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.factorials import subfactorial
from sympy.functions.special.gamma_functions import uppergamma
from sympy.testing.pytest import XFAIL, raises, slow
def test_ff_eval_apply():
    x, y = symbols('x,y')
    n, k = symbols('n k', integer=True)
    m = Symbol('m', integer=True, nonnegative=True)
    assert ff(nan, y) is nan
    assert ff(x, nan) is nan
    assert unchanged(ff, x, y)
    assert ff(oo, 0) == 1
    assert ff(-oo, 0) == 1
    assert ff(oo, 6) is oo
    assert ff(-oo, 7) is -oo
    assert ff(-oo, 6) is oo
    assert ff(oo, -6) is oo
    assert ff(-oo, -7) is oo
    assert ff(x, 0) == 1
    assert ff(x, 1) == x
    assert ff(x, 2) == x * (x - 1)
    assert ff(x, 3) == x * (x - 1) * (x - 2)
    assert ff(x, 5) == x * (x - 1) * (x - 2) * (x - 3) * (x - 4)
    assert ff(x, -1) == 1 / (x + 1)
    assert ff(x, -2) == 1 / ((x + 1) * (x + 2))
    assert ff(x, -3) == 1 / ((x + 1) * (x + 2) * (x + 3))
    assert ff(100, 100) == factorial(100)
    assert ff(2 * x ** 2 - 5 * x, 2) == (2 * x ** 2 - 5 * x) * (2 * x ** 2 - 5 * x - 1)
    assert isinstance(ff(2 * x ** 2 - 5 * x, 2), Mul)
    assert ff(x ** 2 + 3 * x, -2) == 1 / ((x ** 2 + 3 * x + 1) * (x ** 2 + 3 * x + 2))
    assert ff(Poly(2 * x ** 2 - 5 * x, x), 2) == Poly(4 * x ** 4 - 28 * x ** 3 + 59 * x ** 2 - 35 * x, x)
    assert isinstance(ff(Poly(2 * x ** 2 - 5 * x, x), 2), Poly)
    raises(ValueError, lambda: ff(Poly(2 * x ** 2 - 5 * x, x, y), 2))
    assert ff(Poly(x ** 2 + 3 * x, x), -2) == 1 / (x ** 4 + 12 * x ** 3 + 49 * x ** 2 + 78 * x + 40)
    raises(ValueError, lambda: ff(Poly(x ** 2 + 3 * x, x, y), -2))
    assert ff(x, m).is_integer is None
    assert ff(n, k).is_integer is None
    assert ff(n, m).is_integer is True
    assert ff(n, k + pi).is_integer is False
    assert ff(n, m + pi).is_integer is False
    assert ff(pi, m).is_integer is False
    assert isinstance(ff(x, x), ff)
    assert ff(n, n) == factorial(n)

    def check(x, k, o, n):
        a, b = (Dummy(), Dummy())
        r = lambda x, k: o(a, b).rewrite(n).subs({a: x, b: k})
        for i in range(-5, 5):
            for j in range(-5, 5):
                assert o(i, j) == r(i, j), (o, n)
    check(x, k, ff, rf)
    check(x, k, ff, gamma)
    check(n, k, ff, factorial)
    check(x, k, ff, binomial)
    check(x, y, ff, factorial)
    check(x, y, ff, binomial)
    assert ff(x, k).rewrite(rf) == rf(x - k + 1, k)
    assert ff(x, k).rewrite(gamma) == Piecewise((gamma(x + 1) / gamma(-k + x + 1), x >= 0), ((-1) ** k * gamma(k - x) / gamma(-x), True))
    assert ff(5, k).rewrite(gamma) == 120 / gamma(6 - k)
    assert ff(n, k).rewrite(factorial) == Piecewise((factorial(n) / factorial(-k + n), n >= 0), ((-1) ** k * factorial(k - n - 1) / factorial(-n - 1), True))
    assert ff(5, k).rewrite(factorial) == 120 / factorial(5 - k)
    assert ff(x, k).rewrite(binomial) == factorial(k) * binomial(x, k)
    assert ff(x, y).rewrite(factorial) == ff(x, y)
    assert ff(x, y).rewrite(binomial) == ff(x, y)
    import random
    from mpmath import ff as mpmath_ff
    for i in range(100):
        x = -500 + 500 * random.random()
        k = -500 + 500 * random.random()
        a = mpmath_ff(x, k)
        b = ff(x, k)
        assert abs(a - b) < abs(a) * 10 ** (-15)