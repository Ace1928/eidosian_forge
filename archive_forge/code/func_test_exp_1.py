from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
from sympy.core.numbers import (E, I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, XFAIL
def test_exp_1():
    assert exp(x).nseries(x, n=5) == 1 + x + x ** 2 / 2 + x ** 3 / 6 + x ** 4 / 24 + O(x ** 5)
    assert exp(x).nseries(x, n=12) == 1 + x + x ** 2 / 2 + x ** 3 / 6 + x ** 4 / 24 + x ** 5 / 120 + x ** 6 / 720 + x ** 7 / 5040 + x ** 8 / 40320 + x ** 9 / 362880 + x ** 10 / 3628800 + x ** 11 / 39916800 + O(x ** 12)
    assert exp(1 / x).nseries(x, n=5) == exp(1 / x)
    assert exp(1 / (1 + x)).nseries(x, n=4) == (E * (1 - x - 13 * x ** 3 / 6 + 3 * x ** 2 / 2)).expand() + O(x ** 4)
    assert exp(2 + x).nseries(x, n=5) == (exp(2) * (1 + x + x ** 2 / 2 + x ** 3 / 6 + x ** 4 / 24)).expand() + O(x ** 5)