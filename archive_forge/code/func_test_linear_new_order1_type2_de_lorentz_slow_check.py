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
def test_linear_new_order1_type2_de_lorentz_slow_check():
    if ON_CI:
        skip('Too slow for CI.')
    m = Symbol('m', real=True)
    q = Symbol('q', real=True)
    t = Symbol('t', real=True)
    e1, e2, e3 = symbols('e1:4', real=True)
    b1, b2, b3 = symbols('b1:4', real=True)
    v1, v2, v3 = symbols('v1:4', cls=Function, real=True)
    eqs = [-e1 * q + m * Derivative(v1(t), t) - q * (-b2 * v3(t) + b3 * v2(t)), -e2 * q + m * Derivative(v2(t), t) - q * (b1 * v3(t) - b3 * v1(t)), -e3 * q + m * Derivative(v3(t), t) - q * (-b1 * v2(t) + b2 * v1(t))]
    sol = dsolve(eqs)
    assert checksysodesol(eqs, sol) == (True, [0, 0, 0])