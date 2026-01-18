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
def test_gosper_sum_AeqB_part3():
    f3a = 1 / n ** 4
    f3b = (6 * n + 3) / (4 * n ** 4 + 8 * n ** 3 + 8 * n ** 2 + 4 * n + 3)
    f3c = 2 ** n * (n ** 2 - 2 * n - 1) / (n ** 2 * (n + 1) ** 2)
    f3d = n ** 2 * 4 ** n / ((n + 1) * (n + 2))
    f3e = 2 ** n / (n + 1)
    f3f = 4 * (n - 1) * (n ** 2 - 2 * n - 1) / (n ** 2 * (n + 1) ** 2 * (n - 2) ** 2 * (n - 3) ** 2)
    f3g = (n ** 4 - 14 * n ** 2 - 24 * n - 9) * 2 ** n / (n ** 2 * (n + 1) ** 2 * (n + 2) ** 2 * (n + 3) ** 2)
    g3b = m * (m + 2) / (2 * m ** 2 + 4 * m + 3)
    g3c = 2 ** m / m ** 2 - 2
    g3d = Rational(2, 3) + 4 ** (m + 1) * (m - 1) / (m + 2) / 3
    g3f = -(Rational(-1, 16) + 1 / ((m - 2) ** 2 * (m + 1) ** 2))
    g3g = Rational(-2, 9) + 2 ** (m + 1) / ((m + 1) ** 2 * (m + 3) ** 2)
    g = gosper_sum(f3a, (n, 1, m))
    assert g is None
    g = gosper_sum(f3b, (n, 1, m))
    assert g is not None and simplify(g - g3b) == 0
    g = gosper_sum(f3c, (n, 1, m - 1))
    assert g is not None and simplify(g - g3c) == 0
    g = gosper_sum(f3d, (n, 1, m))
    assert g is not None and simplify(g - g3d) == 0
    g = gosper_sum(f3e, (n, 0, m - 1))
    assert g is None
    g = gosper_sum(f3f, (n, 4, m))
    assert g is not None and simplify(g - g3f) == 0
    g = gosper_sum(f3g, (n, 1, m))
    assert g is not None and simplify(g - g3g) == 0