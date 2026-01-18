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
def test_issue_13143():
    f = Function('f')
    fx = f(x).diff(x)
    e = f(x) + fx + f(x) * fx
    assert collect(e, Wild('w')) == f(x) * (fx + 1) + fx
    e = f(x) + f(x) * fx + x * fx * f(x)
    assert collect(e, fx) == (x * f(x) + f(x)) * fx + f(x)
    assert collect(e, f(x)) == (x * fx + fx + 1) * f(x)
    e = f(x) + fx + f(x) * fx
    assert collect(e, [f(x), fx]) == f(x) * (1 + fx) + fx
    assert collect(e, [fx, f(x)]) == fx * (1 + f(x)) + f(x)