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
def test_linear_3eq_order1_type4_slow():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')
    f = t ** 3 + log(t)
    g = t ** 2 + sin(t)
    eq1 = (Eq(diff(x(t), t), (4 * f + g) * x(t) - f * y(t) - 2 * f * z(t)), Eq(diff(y(t), t), 2 * f * x(t) + (f + g) * y(t) - 2 * f * z(t)), Eq(diff(z(t), t), 5 * f * x(t) + f * y(t) + (-3 * f + g) * z(t)))
    with dotprodsimp(True):
        dsolve(eq1)