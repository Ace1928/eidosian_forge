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
def test_geometric_1():
    assert (1 / (1 - x)).nseries(x, n=5) == 1 + x + x ** 2 + x ** 3 + x ** 4 + O(x ** 5)
    assert (x / (1 - x)).nseries(x, n=6) == x + x ** 2 + x ** 3 + x ** 4 + x ** 5 + O(x ** 6)
    assert (x ** 3 / (1 - x)).nseries(x, n=8) == x ** 3 + x ** 4 + x ** 5 + x ** 6 + x ** 7 + O(x ** 8)