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
def test_issue_5849():
    I1, I2, I3, I4, I5, I6 = symbols('I1:7')
    dI1, dI4, dQ2, dQ4, Q2, Q4 = symbols('dI1,dI4,dQ2,dQ4,Q2,Q4')
    e = (I1 - I2 - I3, I3 - I4 - I5, I4 + I5 - I6, -I1 + I2 + I6, -2 * I1 - 2 * I3 - 2 * I5 - 3 * I6 - dI1 / 2 + 12, -I4 + dQ4, -I2 + dQ2, 2 * I3 + 2 * I5 + 3 * I6 - Q2, I4 - 2 * I5 + 2 * Q4 + dI4)
    ans = [{I1: I2 + I3, dI1: -4 * I2 - 8 * I3 - 4 * I5 - 6 * I6 + 24, I4: I3 - I5, dQ4: I3 - I5, Q4: -I3 / 2 + 3 * I5 / 2 - dI4 / 2, dQ2: I2, Q2: 2 * I3 + 2 * I5 + 3 * I6}]
    v = (I1, I4, Q2, Q4, dI1, dI4, dQ2, dQ4)
    assert solve(e, *v, manual=True, check=False, dict=True) == ans
    assert solve(e, *v, manual=True, check=False) == [tuple([a.get(i, i) for i in v]) for a in ans]
    assert solve(e, *v, manual=True) == []
    assert solve(e, *v) == []
    assert [ei.subs(ans[0]) for ei in e] == [0, 0, I3 - I6, -I3 + I6, 0, 0, 0, 0, 0]