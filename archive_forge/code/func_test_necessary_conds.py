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
def test_necessary_conds():
    """
    This function tests the necessary conditions for
    a Riccati ODE to have a rational particular solution.
    """
    assert check_necessary_conds(-3, [1, 2, 4]) == False
    assert check_necessary_conds(1, [1, 2, 4]) == False
    assert check_necessary_conds(2, [3, 1, 6]) == False
    assert check_necessary_conds(-10, [1, 2, 8, 12]) == True