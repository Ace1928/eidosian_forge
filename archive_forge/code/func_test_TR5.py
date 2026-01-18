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
def test_TR5():
    assert TR5(sin(x) ** 2) == -cos(x) ** 2 + 1
    assert TR5(sin(x) ** (-2)) == sin(x) ** (-2)
    assert TR5(sin(x) ** 4) == (-cos(x) ** 2 + 1) ** 2