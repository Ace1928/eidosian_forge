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
def test_TR10():
    assert TR10(cos(a + b)) == -sin(a) * sin(b) + cos(a) * cos(b)
    assert TR10(sin(a + b)) == sin(a) * cos(b) + sin(b) * cos(a)
    assert TR10(sin(a + b + c)) == (-sin(a) * sin(b) + cos(a) * cos(b)) * sin(c) + (sin(a) * cos(b) + sin(b) * cos(a)) * cos(c)
    assert TR10(cos(a + b + c)) == (-sin(a) * sin(b) + cos(a) * cos(b)) * cos(c) - (sin(a) * cos(b) + sin(b) * cos(a)) * sin(c)