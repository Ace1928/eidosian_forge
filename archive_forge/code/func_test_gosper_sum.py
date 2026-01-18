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
def test_gosper_sum():
    assert gosper_sum(1, (k, 0, n)) == 1 + n
    assert gosper_sum(k, (k, 0, n)) == n * (1 + n) / 2
    assert gosper_sum(k ** 2, (k, 0, n)) == n * (1 + n) * (1 + 2 * n) / 6
    assert gosper_sum(k ** 3, (k, 0, n)) == n ** 2 * (1 + n) ** 2 / 4
    assert gosper_sum(2 ** k, (k, 0, n)) == 2 * 2 ** n - 1
    assert gosper_sum(factorial(k), (k, 0, n)) is None
    assert gosper_sum(binomial(n, k), (k, 0, n)) is None
    assert gosper_sum(factorial(k) / k ** 2, (k, 0, n)) is None
    assert gosper_sum((k - 3) * factorial(k), (k, 0, n)) is None
    assert gosper_sum(k * factorial(k), k) == factorial(k)
    assert gosper_sum(k * factorial(k), (k, 0, n)) == n * factorial(n) + factorial(n) - 1
    assert gosper_sum((-1) ** k * binomial(n, k), (k, 0, n)) == 0
    assert gosper_sum((-1) ** k * binomial(n, k), (k, 0, m)) == -(-1) ** m * (m - n) * binomial(n, m) / n
    assert gosper_sum((4 * k + 1) * factorial(k) / factorial(2 * k + 1), (k, 0, n)) == (2 * factorial(2 * n + 1) - factorial(n)) / factorial(2 * n + 1)
    assert gosper_sum(n * (n + a + b) * a ** n * b ** n / (factorial(n + a) * factorial(n + b)), (n, 0, m)).simplify() == -exp(m * log(a) + m * log(b)) * gamma(a + 1) * gamma(b + 1) / (gamma(a) * gamma(b) * gamma(a + m + 1) * gamma(b + m + 1)) + 1 / (gamma(a) * gamma(b))