from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify
from sympy.concrete.gosper import gosper_normal, gosper_sum, gosper_term
from sympy.abc import a, b, j, k, m, n, r, x
def test_gosper_nan():
    a = Symbol('a', positive=True)
    b = Symbol('b', positive=True)
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True)
    f2d = n * (n + a + b) * a ** n * b ** n / (factorial(n + a) * factorial(n + b))
    g2d = 1 / (factorial(a - 1) * factorial(b - 1)) - a ** (m + 1) * b ** (m + 1) / (factorial(a + m) * factorial(b + m))
    g = gosper_sum(f2d, (n, 0, m))
    assert simplify(g - g2d) == 0