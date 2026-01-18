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
def test_collect_order():
    a, b, x, t = symbols('a,b,x,t')
    assert collect(t + t * x + t * x ** 2 + O(x ** 3), t) == t * (1 + x + x ** 2 + O(x ** 3))
    assert collect(t + t * x + x ** 2 + O(x ** 3), t) == t * (1 + x + O(x ** 3)) + x ** 2 + O(x ** 3)
    f = a * x + b * x + c * x ** 2 + d * x ** 2 + O(x ** 3)
    g = x * (a + b) + x ** 2 * (c + d) + O(x ** 3)
    assert collect(f, x) == g
    assert collect(f, x, distribute_order_term=False) == g
    f = sin(a + b).series(b, 0, 10)
    assert collect(f, [sin(a), cos(a)]) == sin(a) * cos(b).series(b, 0, 10) + cos(a) * sin(b).series(b, 0, 10)
    assert collect(f, [sin(a), cos(a)], distribute_order_term=False) == sin(a) * cos(b).series(b, 0, 10).removeO() + cos(a) * sin(b).series(b, 0, 10).removeO() + O(b ** 10)