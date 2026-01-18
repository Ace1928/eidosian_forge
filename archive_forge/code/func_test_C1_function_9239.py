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
def test_C1_function_9239():
    t = Symbol('t')
    C1 = Function('C1')
    C2 = Function('C2')
    C3 = Symbol('C3')
    C4 = Symbol('C4')
    eq = (Eq(diff(C1(t), t), 9 * C2(t)), Eq(diff(C2(t), t), 12 * C1(t)))
    sol = [Eq(C1(t), 9 * C3 * exp(6 * sqrt(3) * t) + 9 * C4 * exp(-6 * sqrt(3) * t)), Eq(C2(t), 6 * sqrt(3) * C3 * exp(6 * sqrt(3) * t) - 6 * sqrt(3) * C4 * exp(-6 * sqrt(3) * t))]
    assert checksysodesol(eq, sol) == (True, [0, 0])