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
def test__TR11():
    assert _TR11(sin(x / 3) * sin(2 * x) * sin(x / 4) / (cos(x / 6) * cos(x / 8))) == 4 * sin(x / 8) * sin(x / 6) * sin(2 * x), _TR11(sin(x / 3) * sin(2 * x) * sin(x / 4) / (cos(x / 6) * cos(x / 8)))
    assert _TR11(sin(x / 3) / cos(x / 6)) == 2 * sin(x / 6)
    assert _TR11(cos(x / 6) / sin(x / 3)) == 1 / (2 * sin(x / 6))
    assert _TR11(sin(2 * x) * cos(x / 8) / sin(x / 4)) == sin(2 * x) / (2 * sin(x / 8)), _TR11(sin(2 * x) * cos(x / 8) / sin(x / 4))
    assert _TR11(sin(x) / sin(x / 2)) == 2 * cos(x / 2)