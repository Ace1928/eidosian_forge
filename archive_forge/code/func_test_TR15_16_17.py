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
def test_TR15_16_17():
    assert TR15(1 - 1 / sin(x) ** 2) == -cot(x) ** 2
    assert TR16(1 - 1 / cos(x) ** 2) == -tan(x) ** 2
    assert TR111(1 - 1 / tan(x) ** 2) == 1 - cot(x) ** 2