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
def test_gosper_term():
    assert gosper_term((4 * k + 1) * factorial(k) / factorial(2 * k + 1), k) == (-k - S.Half) / (k + Rational(1, 4))