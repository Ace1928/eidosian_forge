from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
def test_Limits_simple_2():
    assert limit(1000 * x / (x ** 2 - 1), x, oo) == 0
    assert limit((x ** 2 - 5 * x + 1) / (3 * x + 7), x, oo) is oo
    assert limit((2 * x ** 2 - x + 3) / (x ** 3 - 8 * x + 5), x, oo) == 0
    assert limit((2 * x ** 2 - 3 * x - 4) / sqrt(x ** 4 + 1), x, oo) == 2
    assert limit((2 * x + 3) / (x + root3(x)), x, oo) == 2
    assert limit(x ** 2 / (10 + x * sqrt(x)), x, oo) is oo
    assert limit(root3(x ** 2 + 1) / (x + 1), x, oo) == 0
    assert limit(sqrt(x) / sqrt(x + sqrt(x + sqrt(x))), x, oo) == 1