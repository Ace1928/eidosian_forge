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
def test_TR3():
    assert TR3(cos(y - x * (y - x))) == cos(x * (x - y) + y)
    assert cos(pi / 2 + x) == -sin(x)
    assert cos(30 * pi / 2 + x) == -cos(x)
    for f in (cos, sin, tan, cot, csc, sec):
        i = f(pi * Rational(3, 7))
        j = TR3(i)
        assert verify_numerically(i, j) and i.func != j.func