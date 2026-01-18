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
def test_gosper_normal():
    eq = (4 * n + 5, 2 * (4 * n + 1) * (2 * n + 3), n)
    assert gosper_normal(*eq) == (Poly(Rational(1, 4), n), Poly(n + Rational(3, 2)), Poly(n + Rational(1, 4)))
    assert gosper_normal(*eq, polys=False) == (Rational(1, 4), n + Rational(3, 2), n + Rational(1, 4))