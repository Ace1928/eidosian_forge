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
def test_collect_D():
    D = Derivative
    f = Function('f')
    x, a, b = symbols('x,a,b')
    fx = D(f(x), x)
    fxx = D(f(x), x, x)
    assert collect(a * fx + b * fx, fx) == (a + b) * fx
    assert collect(a * D(fx, x) + b * D(fx, x), fx) == (a + b) * D(fx, x)
    assert collect(a * fxx + b * fxx, fx) == (a + b) * D(fx, x)
    assert collect(5 * f(x) + 3 * fx, fx) == 5 * f(x) + 3 * fx
    assert collect(f(x) + f(x) * diff(f(x), x) + x * diff(f(x), x) * f(x), f(x).diff(x)) == (x * f(x) + f(x)) * D(f(x), x) + f(x)
    assert collect(f(x) + f(x) * diff(f(x), x) + x * diff(f(x), x) * f(x), f(x).diff(x), exact=True) == (x * f(x) + f(x)) * D(f(x), x) + f(x)
    assert collect(1 / f(x) + 1 / f(x) * diff(f(x), x) + x * diff(f(x), x) / f(x), f(x).diff(x), exact=True) == (1 / f(x) + x / f(x)) * D(f(x), x) + 1 / f(x)
    e = (1 + x * fx + fx) / f(x)
    assert collect(e.expand(), fx) == fx * (x / f(x) + 1 / f(x)) + 1 / f(x)