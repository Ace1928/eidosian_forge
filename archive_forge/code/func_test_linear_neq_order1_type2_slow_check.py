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
def test_linear_neq_order1_type2_slow_check():
    RC, t, C, Vs, L, R1, V0, I0 = symbols('RC t C Vs L R1 V0 I0')
    V = Function('V')
    I = Function('I')
    system = [Eq(V(t).diff(t), -1 / RC * V(t) + I(t) / C), Eq(I(t).diff(t), -R1 / L * I(t) - 1 / L * V(t) + Vs / L)]
    [sol] = dsolve_system(system, simplify=False, doit=False)
    assert checksysodesol(system, sol) == (True, [0, 0])