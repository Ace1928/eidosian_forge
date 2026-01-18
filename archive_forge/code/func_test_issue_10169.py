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
def test_issue_10169():
    eq = S(-8 * a - x ** 5 * (a + b + c + e) - x ** 4 * (4 * a - 2 ** Rational(3, 4) * c + 4 * c + d + 2 ** Rational(3, 4) * e + 4 * e + k) - x ** 3 * (-4 * 2 ** Rational(3, 4) * c + sqrt(2) * c - 2 ** Rational(3, 4) * d + 4 * d + sqrt(2) * e + 4 * 2 ** Rational(3, 4) * e + 2 ** Rational(3, 4) * k + 4 * k) - x ** 2 * (4 * sqrt(2) * c - 4 * 2 ** Rational(3, 4) * d + sqrt(2) * d + 4 * sqrt(2) * e + sqrt(2) * k + 4 * 2 ** Rational(3, 4) * k) - x * (2 * a + 2 * b + 4 * sqrt(2) * d + 4 * sqrt(2) * k) + 5)
    assert solve_undetermined_coeffs(eq, [a, b, c, d, e, k], x) == {a: Rational(5, 8), b: Rational(-5, 1032), c: Rational(-40, 129) - 5 * 2 ** Rational(3, 4) / 129 + 5 * 2 ** Rational(1, 4) / 1032, d: -20 * 2 ** Rational(3, 4) / 129 - 10 * sqrt(2) / 129 - 5 * 2 ** Rational(1, 4) / 258, e: Rational(-40, 129) - 5 * 2 ** Rational(1, 4) / 1032 + 5 * 2 ** Rational(3, 4) / 129, k: -10 * sqrt(2) / 129 + 5 * 2 ** Rational(1, 4) / 258 + 20 * 2 ** Rational(3, 4) / 129}