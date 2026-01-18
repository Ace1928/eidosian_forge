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
def test_issue_4671_4463_4467():
    assert solve(sqrt(x ** 2 - 1) - 2) in ([sqrt(5), -sqrt(5)], [-sqrt(5), sqrt(5)])
    assert solve((2 ** exp(y ** 2 / x) + 2) / (x ** 2 + 15), y) == [-sqrt(x * log(1 + I * pi / log(2))), sqrt(x * log(1 + I * pi / log(2)))]
    C1, C2 = symbols('C1 C2')
    f = Function('f')
    assert solve(C1 + C2 / x ** 2 - exp(-f(x)), f(x)) == [log(x ** 2 / (C1 * x ** 2 + C2))]
    a = Symbol('a')
    E = S.Exp1
    assert solve(1 - log(a + 4 * x ** 2), x) in ([-sqrt(-a + E) / 2, sqrt(-a + E) / 2], [sqrt(-a + E) / 2, -sqrt(-a + E) / 2])
    assert solve(log(a ** (-3) - x ** 2) / a, x) in ([-sqrt(-1 + a ** (-3)), sqrt(-1 + a ** (-3))], [sqrt(-1 + a ** (-3)), -sqrt(-1 + a ** (-3))])
    assert solve(1 - log(a + 4 * x ** 2), x) in ([-sqrt(-a + E) / 2, sqrt(-a + E) / 2], [sqrt(-a + E) / 2, -sqrt(-a + E) / 2])
    assert solve((a ** 2 + 1) * (sin(a * x) + cos(a * x)), x) == [-pi / (4 * a)]
    assert solve(3 - (sinh(a * x) + cosh(a * x)), x) == [log(3) / a]
    assert set(solve(3 - (sinh(a * x) + cosh(a * x) ** 2), x)) == {log(-2 + sqrt(5)) / a, log(-sqrt(2) + 1) / a, log(-sqrt(5) - 2) / a, log(1 + sqrt(2)) / a}
    assert solve(atan(x) - 1) == [tan(1)]