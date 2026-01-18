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
def test_difference_delta__Pow():
    e = 4 ** n
    assert dd(e, n) == 3 * 4 ** n
    assert dd(e, n, 2) == 15 * 4 ** n
    e = 4 ** (2 * n)
    assert dd(e, n) == 15 * 4 ** (2 * n)
    assert dd(e, n, 2) == 255 * 4 ** (2 * n)
    e = n ** 4
    assert dd(e, n) == (n + 1) ** 4 - n ** 4
    e = n ** n
    assert dd(e, n) == (n + 1) ** (n + 1) - n ** n