from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_checkpdesol():
    f, F = map(Function, ['f', 'F'])
    eq1 = a * f(x, y) + b * f(x, y).diff(x) + c * f(x, y).diff(y)
    eq2 = 3 * f(x, y) + 2 * f(x, y).diff(x) + f(x, y).diff(y)
    eq3 = a * f(x, y) + b * f(x, y).diff(x) + 2 * f(x, y).diff(y)
    for eq in [eq1, eq2, eq3]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    eq4 = x * f(x, y) + f(x, y).diff(x) + 3 * f(x, y).diff(y)
    eq5 = 2 * f(x, y) + 1 * f(x, y).diff(x) + 3 * f(x, y).diff(y)
    eq6 = f(x, y) + 1 * f(x, y).diff(x) + 3 * f(x, y).diff(y)
    assert checkpdesol(eq4, [pdsolve(eq5), pdsolve(eq6)]) == [(False, (x - 2) * F(3 * x - y) * exp(-x / S(5) - 3 * y / S(5))), (False, (x - 1) * F(3 * x - y) * exp(-x / S(10) - 3 * y / S(10)))]
    for eq in [eq4, eq5, eq6]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    sol = pdsolve(eq4)
    sol4 = Eq(sol.lhs - sol.rhs, 0)
    raises(NotImplementedError, lambda: checkpdesol(eq4, sol4, solve_for_func=False))