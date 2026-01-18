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
def test__TR56():
    h = lambda x: 1 - x
    assert T(sin(x) ** 3, sin, cos, h, 4, False) == sin(x) * (-cos(x) ** 2 + 1)
    assert T(sin(x) ** 10, sin, cos, h, 4, False) == sin(x) ** 10
    assert T(sin(x) ** 6, sin, cos, h, 6, False) == (-cos(x) ** 2 + 1) ** 3
    assert T(sin(x) ** 6, sin, cos, h, 6, True) == sin(x) ** 6
    assert T(sin(x) ** 8, sin, cos, h, 10, True) == (-cos(x) ** 2 + 1) ** 4
    assert T(sin(x) ** I, sin, cos, h, 4, True) == sin(x) ** I
    assert T(sin(x) ** (2 * I + 1), sin, cos, h, 4, True) == sin(x) ** (2 * I + 1)