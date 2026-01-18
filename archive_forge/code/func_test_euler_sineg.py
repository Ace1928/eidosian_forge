from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.calculus.euler import euler_equations as euler
def test_euler_sineg():
    psi = Function('psi')
    t = Symbol('t')
    x = Symbol('x')
    L = D(psi(t, x), t) ** 2 / 2 - D(psi(t, x), x) ** 2 / 2 + cos(psi(t, x))
    assert euler(L, psi(t, x), [t, x]) == [Eq(-sin(psi(t, x)) - D(psi(t, x), t, t) + D(psi(t, x), x, x), 0)]