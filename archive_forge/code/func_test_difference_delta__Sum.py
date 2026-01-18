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
def test_difference_delta__Sum():
    e = Sum(1 / k, (k, 1, n))
    assert dd(e, n) == 1 / (n + 1)
    assert dd(e, n, 5) == Add(*[1 / (i + n + 1) for i in range(5)])
    e = Sum(1 / k, (k, 1, 3 * n))
    assert dd(e, n) == Add(*[1 / (i + 3 * n + 1) for i in range(3)])
    e = n * Sum(1 / k, (k, 1, n))
    assert dd(e, n) == 1 + Sum(1 / k, (k, 1, n))
    e = Sum(1 / k, (k, 1, n), (m, 1, n))
    assert dd(e, n) == harmonic(n)