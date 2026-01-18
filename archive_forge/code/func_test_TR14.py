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
def test_TR14():
    eq = (cos(x) - 1) * (cos(x) + 1)
    ans = -sin(x) ** 2
    assert TR14(eq) == ans
    assert TR14(1 / eq) == 1 / ans
    assert TR14((cos(x) - 1) ** 2 * (cos(x) + 1) ** 2) == ans ** 2
    assert TR14((cos(x) - 1) ** 2 * (cos(x) + 1) ** 3) == ans ** 2 * (cos(x) + 1)
    assert TR14((cos(x) - 1) ** 3 * (cos(x) + 1) ** 2) == ans ** 2 * (cos(x) - 1)
    eq = (cos(x) - 1) ** y * (cos(x) + 1) ** y
    assert TR14(eq) == eq
    eq = (cos(x) - 2) ** y * (cos(x) + 1)
    assert TR14(eq) == eq
    eq = (tan(x) - 2) ** 2 * (cos(x) + 1)
    assert TR14(eq) == eq
    i = symbols('i', integer=True)
    assert TR14((cos(x) - 1) ** i * (cos(x) + 1) ** i) == ans ** i
    assert TR14((sin(x) - 1) ** i * (sin(x) + 1) ** i) == (-cos(x) ** 2) ** i
    eq = (cos(x) - 1) ** (i + 1) * (cos(x) + 1) ** i
    assert TR14(eq) in [(cos(x) - 1) * ans ** i, eq]
    assert TR14((sin(x) - 1) * (sin(x) + 1)) == -cos(x) ** 2
    p1 = (cos(x) + 1) * (cos(x) - 1)
    p2 = (cos(y) - 1) * 2 * (cos(y) + 1)
    p3 = 3 * (cos(y) - 1) * (3 * (cos(y) + 1))
    assert TR14(p1 * p2 * p3 * (x - 1)) == -18 * ((x - 1) * sin(x) ** 2 * sin(y) ** 4)