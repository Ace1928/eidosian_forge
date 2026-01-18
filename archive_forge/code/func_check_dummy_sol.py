from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
def check_dummy_sol(eq, solse, dummy_sym):
    """
    Helper function to check if actual solution
    matches expected solution if actual solution
    contains dummy symbols.
    """
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    _, funcs = match_riccati(eq, f, x)
    sols = solve_riccati(f(x), x, *funcs)
    C1 = Dummy('C1')
    sols = [sol.subs(C1, dummy_sym) for sol in sols]
    assert all([x[0] for x in checkodesol(eq, sols)])
    assert all([s1.dummy_eq(s2, dummy_sym) for s1, s2 in zip(sols, solse)])