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
@slow
def test_issue_2840_8155():
    assert solve(sin(3 * x) + sin(6 * x)) == [0, pi * Rational(-5, 3), pi * Rational(-4, 3), -pi, pi * Rational(-2, 3), pi * Rational(-4, 9), -pi / 3, pi * Rational(-2, 9), pi * Rational(2, 9), pi / 3, pi * Rational(4, 9), pi * Rational(2, 3), pi, pi * Rational(4, 3), pi * Rational(14, 9), pi * Rational(5, 3), pi * Rational(16, 9), 2 * pi, -2 * I * log(-(-1) ** Rational(1, 9)), -2 * I * log(-(-1) ** Rational(2, 9)), -2 * I * log(-sin(pi / 18) - I * cos(pi / 18)), -2 * I * log(-sin(pi / 18) + I * cos(pi / 18)), -2 * I * log(sin(pi / 18) - I * cos(pi / 18)), -2 * I * log(sin(pi / 18) + I * cos(pi / 18))]
    assert solve(2 * sin(x) - 2 * sin(2 * x)) == [0, pi * Rational(-5, 3), -pi, -pi / 3, pi / 3, pi, pi * Rational(5, 3)]