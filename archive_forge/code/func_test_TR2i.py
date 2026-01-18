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
def test_TR2i():
    assert powsimp(2 ** x / y ** x) != (2 / y) ** x
    assert TR2i(sin(x) / cos(x)) == tan(x)
    assert TR2i(sin(x) * sin(y) / cos(x)) == tan(x) * sin(y)
    assert TR2i(1 / (sin(x) / cos(x))) == 1 / tan(x)
    assert TR2i(1 / (sin(x) * sin(y) / cos(x))) == 1 / tan(x) / sin(y)
    assert TR2i(sin(x) / 2 / (cos(x) + 1)) == sin(x) / (cos(x) + 1) / 2
    assert TR2i(sin(x) / 2 / (cos(x) + 1), half=True) == tan(x / 2) / 2
    assert TR2i(sin(1) / (cos(1) + 1), half=True) == tan(S.Half)
    assert TR2i(sin(2) / (cos(2) + 1), half=True) == tan(1)
    assert TR2i(sin(4) / (cos(4) + 1), half=True) == tan(2)
    assert TR2i(sin(5) / (cos(5) + 1), half=True) == tan(5 * S.Half)
    assert TR2i((cos(1) + 1) / sin(1), half=True) == 1 / tan(S.Half)
    assert TR2i((cos(2) + 1) / sin(2), half=True) == 1 / tan(1)
    assert TR2i((cos(4) + 1) / sin(4), half=True) == 1 / tan(2)
    assert TR2i((cos(5) + 1) / sin(5), half=True) == 1 / tan(5 * S.Half)
    assert TR2i((cos(1) + 1) ** (-a) * sin(1) ** a, half=True) == tan(S.Half) ** a
    assert TR2i((cos(2) + 1) ** (-a) * sin(2) ** a, half=True) == tan(1) ** a
    assert TR2i((cos(4) + 1) ** (-a) * sin(4) ** a, half=True) == (cos(4) + 1) ** (-a) * sin(4) ** a
    assert TR2i((cos(5) + 1) ** (-a) * sin(5) ** a, half=True) == (cos(5) + 1) ** (-a) * sin(5) ** a
    assert TR2i((cos(1) + 1) ** a * sin(1) ** (-a), half=True) == tan(S.Half) ** (-a)
    assert TR2i((cos(2) + 1) ** a * sin(2) ** (-a), half=True) == tan(1) ** (-a)
    assert TR2i((cos(4) + 1) ** a * sin(4) ** (-a), half=True) == (cos(4) + 1) ** a * sin(4) ** (-a)
    assert TR2i((cos(5) + 1) ** a * sin(5) ** (-a), half=True) == (cos(5) + 1) ** a * sin(5) ** (-a)
    i = symbols('i', integer=True)
    assert TR2i((cos(5) + 1) ** i * sin(5) ** (-i), half=True) == tan(5 * S.Half) ** (-i)
    assert TR2i(1 / ((cos(5) + 1) ** i * sin(5) ** (-i)), half=True) == tan(5 * S.Half) ** i