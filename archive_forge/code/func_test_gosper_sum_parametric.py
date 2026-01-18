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
def test_gosper_sum_parametric():
    assert gosper_sum(binomial(S.Half, m - j + 1) * binomial(S.Half, m + j), (j, 1, n)) == n * (1 + m - n) * (-1 + 2 * m + 2 * n) * binomial(S.Half, 1 + m - n) * binomial(S.Half, m + n) / (m * (1 + 2 * m))