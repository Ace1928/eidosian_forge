from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
def test_f1b():
    m = Symbol('m')
    n = Symbol('n')
    h = Symbol('h')
    a = Symbol('a')
    assert limit(sin(x) / x, x, 2) == sin(2) / 2
    assert limit(sin(3 * x) / x, x, 0) == 3
    assert limit(sin(5 * x) / sin(2 * x), x, 0) == Rational(5, 2)
    assert limit(sin(pi * x) / sin(3 * pi * x), x, 0) == Rational(1, 3)
    assert limit(x * sin(pi / x), x, oo) == pi
    assert limit((1 - cos(x)) / x ** 2, x, 0) == S.Half
    assert limit(x * sin(1 / x), x, oo) == 1
    assert limit((cos(m * x) - cos(n * x)) / x ** 2, x, 0) == -m ** 2 / 2 + n ** 2 / 2
    assert limit((tan(x) - sin(x)) / x ** 3, x, 0) == S.Half
    assert limit((x - sin(2 * x)) / (x + sin(3 * x)), x, 0) == -Rational(1, 4)
    assert limit((1 - sqrt(cos(x))) / x ** 2, x, 0) == Rational(1, 4)
    assert limit((sqrt(1 + sin(x)) - sqrt(1 - sin(x))) / x, x, 0) == 1
    assert limit((1 + h / x) ** x, x, oo) == exp(h)
    assert limit((sin(x) - sin(a)) / (x - a), x, a) == cos(a)
    assert limit((cos(x) - cos(a)) / (x - a), x, a) == -sin(a)
    assert limit((sin(x + h) - sin(x)) / h, h, 0) == cos(x)