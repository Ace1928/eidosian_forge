from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.combinatorial.numbers import (fibonacci, harmonic)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.series.limitseq import limit_seq
from sympy.series.limitseq import difference_delta as dd
from sympy.testing.pytest import raises, XFAIL
from sympy.calculus.accumulationbounds import AccumulationBounds
@XFAIL
def test_limit_seq_fail():
    e = harmonic(n) ** 3 * Sum(1 / harmonic(k), (k, 1, n)) / (n * Sum(harmonic(k) / k, (k, 1, n)))
    assert limit_seq(e, n) == 2
    e = Sum(2 ** k * binomial(2 * k, k) / k ** 2, (k, 1, n)) / (Sum(2 ** k / k * 2, (k, 1, n)) * Sum(binomial(2 * k, k), (k, 1, n)))
    assert limit_seq(e, n) == S(3) / 7
    e = n ** 3 * Sum(2 ** k / k ** 2, (k, 1, n)) ** 2 / (2 ** n * Sum(2 ** k / k, (k, 1, n)))
    assert limit_seq(e, n) == 2
    e = harmonic(n) * Sum(2 ** k / k, (k, 1, n)) / (n * Sum(2 ** k * harmonic(k) / k ** 2, (k, 1, n)))
    assert limit_seq(e, n) == 1
    e = Sum(2 ** k * factorial(k) / k ** 2, (k, 1, 2 * n)) / (Sum(4 ** k / k ** 2, (k, 1, n)) * Sum(factorial(k), (k, 1, 2 * n)))
    assert limit_seq(e, n) == S(3) / 16