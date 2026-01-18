from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.series.order import O
from sympy.simplify.radsimp import (collect, collect_const, fraction, radsimp, rcollect)
from sympy.core.expr import unchanged
from sympy.core.mul import _unevaluated_Mul as umul
from sympy.simplify.radsimp import (_unevaluated_Add,
from sympy.testing.pytest import raises
from sympy.abc import x, y, z, a, b, c, d
def test_collect_3():
    """Collect with respect to a product"""
    a, b, c = symbols('a,b,c')
    f = Function('f')
    x, y, z, n = symbols('x,y,z,n')
    assert collect(-x / 8 + x * y, -x) == x * (y - Rational(1, 8))
    assert collect(1 + x * y ** 2, x * y) == 1 + x * y ** 2
    assert collect(x * y + a * x * y, x * y) == x * y * (1 + a)
    assert collect(1 + x * y + a * x * y, x * y) == 1 + x * y * (1 + a)
    assert collect(a * x * f(x) + b * (x * f(x)), x * f(x)) == x * (a + b) * f(x)
    assert collect(a * x * log(x) + b * (x * log(x)), x * log(x)) == x * (a + b) * log(x)
    assert collect(a * x ** 2 * log(x) ** 2 + b * (x * log(x)) ** 2, x * log(x)) == x ** 2 * log(x) ** 2 * (a + b)
    assert collect(y * x * z + a * x * y * z, x * y * z) == (1 + a) * x * y * z