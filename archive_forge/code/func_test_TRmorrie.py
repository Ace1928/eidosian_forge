from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.hyperbolic import (cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.simplify.powsimp import powsimp
from sympy.simplify.fu import (
from sympy.core.random import verify_numerically
from sympy.abc import a, b, c, x, y, z
def test_TRmorrie():
    assert TRmorrie(7 * Mul(*[cos(i) for i in range(10)])) == 7 * sin(12) * sin(16) * cos(5) * cos(7) * cos(9) / (64 * sin(1) * sin(3))
    assert TRmorrie(x) == x
    assert TRmorrie(2 * x) == 2 * x
    e = cos(pi / 7) * cos(pi * Rational(2, 7)) * cos(pi * Rational(4, 7))
    assert TR8(TRmorrie(e)) == Rational(-1, 8)
    e = Mul(*[cos(2 ** i * pi / 17) for i in range(1, 17)])
    assert TR8(TR3(TRmorrie(e))) == Rational(1, 65536)
    eq = cos(x) / cos(x / 2)
    assert TRmorrie(eq) == eq
    eq = cos(x / 2) * sin(x / 2) * cos(x) ** 3
    assert TRmorrie(eq) == sin(2 * x) * cos(x) ** 2 / 4