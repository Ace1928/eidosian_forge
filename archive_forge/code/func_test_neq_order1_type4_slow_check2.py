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
@slow
def test_neq_order1_type4_slow_check2():
    f, g, h = symbols('f, g, h', cls=Function)
    x = Symbol('x')
    eqs = [Eq(Derivative(f(x), x), x * h(x) + f(x) + g(x) + 1), Eq(Derivative(g(x), x), x * g(x) + f(x) + h(x) + 10), Eq(Derivative(h(x), x), x * f(x) + x + g(x) + h(x))]
    with dotprodsimp(True):
        sol = dsolve(eqs)
    assert checksysodesol(eqs, sol) == (True, [0, 0, 0])