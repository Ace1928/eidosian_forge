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
def test_TR12():
    assert TR12(tan(x + y)) == (tan(x) + tan(y)) / (-tan(x) * tan(y) + 1)
    assert TR12(tan(x + y + z)) == (tan(z) + (tan(x) + tan(y)) / (-tan(x) * tan(y) + 1)) / (1 - (tan(x) + tan(y)) * tan(z) / (-tan(x) * tan(y) + 1))
    assert TR12(tan(x * y)) == tan(x * y)