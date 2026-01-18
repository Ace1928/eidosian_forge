from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import Poly
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)
from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn
from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R
def test_issue_14607():
    s, tau_c, tau_1, tau_2, phi, K = symbols('s, tau_c, tau_1, tau_2, phi, K')
    target = (s ** 2 * tau_1 * tau_2 + s * tau_1 + s * tau_2 + 1) / (K * s * (-phi + tau_c))
    K_C, tau_I, tau_D = symbols('K_C, tau_I, tau_D', positive=True, nonzero=True)
    PID = K_C * (1 + 1 / (tau_I * s) + tau_D * s)
    eq = (target - PID).together()
    eq *= denom(eq).simplify()
    eq = Poly(eq, s)
    c = eq.coeffs()
    vars = [K_C, tau_I, tau_D]
    s = solve(c, vars, dict=True)
    assert len(s) == 1
    knownsolution = {K_C: -(tau_1 + tau_2) / (K * (phi - tau_c)), tau_I: tau_1 + tau_2, tau_D: tau_1 * tau_2 / (tau_1 + tau_2)}
    for var in vars:
        assert s[0][var].simplify() == knownsolution[var].simplify()