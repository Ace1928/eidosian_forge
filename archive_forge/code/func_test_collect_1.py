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
def test_collect_1():
    """Collect with respect to Symbol"""
    x, y, z, n = symbols('x,y,z,n')
    assert collect(1, x) == 1
    assert collect(x + y * x, x) == x * (1 + y)
    assert collect(x + x ** 2, x) == x + x ** 2
    assert collect(x ** 2 + y * x ** 2, x) == x ** 2 * (1 + y)
    assert collect(x ** 2 + y * x, x) == x * y + x ** 2
    assert collect(2 * x ** 2 + y * x ** 2 + 3 * x * y, [x]) == x ** 2 * (2 + y) + 3 * x * y
    assert collect(2 * x ** 2 + y * x ** 2 + 3 * x * y, [y]) == 2 * x ** 2 + y * (x ** 2 + 3 * x)
    assert collect(((1 + y + x) ** 4).expand(), x) == ((1 + y) ** 4).expand() + x * (4 * (1 + y) ** 3).expand() + x ** 2 * (6 * (1 + y) ** 2).expand() + x ** 3 * (4 * (1 + y)).expand() + x ** 4
    expr = x + y
    assert collect(expr, expr.free_symbols) == expr
    assert collect(x * exp(x) + sin(x) * y + sin(x) * 2 + 3 * x, x, exact=None) == x * exp(x) + 3 * x + (y + 2) * sin(x)
    assert collect(x * exp(x) + sin(x) * y + sin(x) * 2 + 3 * x + y * x + y * x * exp(x), x, exact=None) == x * exp(x) * (y + 1) + (3 + y) * x + (y + 2) * sin(x)