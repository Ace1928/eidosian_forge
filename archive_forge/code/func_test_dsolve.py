from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
def test_dsolve():
    f, g = symbols('f g', cls=Function)
    x, y = symbols('x y')
    eqs = [f(x).diff(x) - x, f(x).diff(x) + x]
    with raises(ValueError):
        dsolve(eqs)
    eqs = [f(x, y).diff(x)]
    with raises(ValueError):
        dsolve(eqs)
    eqs = [f(x, y).diff(x) + g(x).diff(x), g(x).diff(x)]
    with raises(ValueError):
        dsolve(eqs)