from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
def test_Limits_simple_1():
    assert limit((x + 1) * (x + 2) * (x + 3) / x ** 3, x, oo) == 1
    assert limit(sqrt(x + 1) - sqrt(x), x, oo) == 0
    assert limit((2 * x - 3) * (3 * x + 5) * (4 * x - 6) / (3 * x ** 3 + x - 1), x, oo) == 8
    assert limit(x / root3(x ** 3 + 10), x, oo) == 1
    assert limit((x + 1) ** 2 / (x ** 2 + 1), x, oo) == 1