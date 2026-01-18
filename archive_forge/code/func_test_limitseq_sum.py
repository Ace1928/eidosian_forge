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
def test_limitseq_sum():
    from sympy.abc import x, y, z
    assert limit_seq(Sum(1 / x, (x, 1, y)) - log(y), y) == S.EulerGamma
    assert limit_seq(Sum(1 / x, (x, 1, y)) - 1 / y, y) is S.Infinity
    assert limit_seq(binomial(2 * x, x) / Sum(binomial(2 * y, y), (y, 1, x)), x) == S(3) / 4
    assert limit_seq(Sum(y ** 2 * Sum(2 ** z / z, (z, 1, y)), (y, 1, x)) / (2 ** x * x), x) == 4