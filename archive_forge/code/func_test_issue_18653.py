from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.calculus.euler import euler_equations as euler
def test_issue_18653():
    x, y, z = symbols('x y z')
    f, g, h = symbols('f g h', cls=Function, args=(x, y))
    f, g, h = (f(), g(), h())
    expr2 = f.diff(x) * h.diff(z)
    assert euler(expr2, (f,), (x, y)) == []