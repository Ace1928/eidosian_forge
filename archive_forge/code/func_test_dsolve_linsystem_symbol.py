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
def test_dsolve_linsystem_symbol():
    eps = Symbol('epsilon', positive=True)
    eq1 = (Eq(diff(f(x), x), -eps * g(x)), Eq(diff(g(x), x), eps * f(x)))
    sol1 = [Eq(f(x), -C1 * eps * cos(eps * x) - C2 * eps * sin(eps * x)), Eq(g(x), -C1 * eps * sin(eps * x) + C2 * eps * cos(eps * x))]
    assert checksysodesol(eq1, sol1) == (True, [0, 0])