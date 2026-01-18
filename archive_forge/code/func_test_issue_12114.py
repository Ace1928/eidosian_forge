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
def test_issue_12114():
    a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g')
    terms = [1 + a * b + d * e, 1 + a * c + d * f, 1 + b * c + e * f, g - a ** 2 - d ** 2, g - b ** 2 - e ** 2, g - c ** 2 - f ** 2]
    sol = solve(terms, [a, b, c, d, e, f, g], dict=True)
    s = sqrt(-f ** 2 - 1)
    s2 = sqrt(2 - f ** 2)
    s3 = sqrt(6 - 3 * f ** 2)
    s4 = sqrt(3) * f
    s5 = sqrt(3) * s2
    assert sol == [{a: -s, b: -s, c: -s, d: f, e: f, g: -1}, {a: s, b: s, c: s, d: f, e: f, g: -1}, {a: -s4 / 2 - s2 / 2, b: s4 / 2 - s2 / 2, c: s2, d: -f / 2 + s3 / 2, e: -f / 2 - s5 / 2, g: 2}, {a: -s4 / 2 + s2 / 2, b: s4 / 2 + s2 / 2, c: -s2, d: -f / 2 - s3 / 2, e: -f / 2 + s5 / 2, g: 2}, {a: s4 / 2 - s2 / 2, b: -s4 / 2 - s2 / 2, c: s2, d: -f / 2 - s3 / 2, e: -f / 2 + s5 / 2, g: 2}, {a: s4 / 2 + s2 / 2, b: -s4 / 2 + s2 / 2, c: -s2, d: -f / 2 + s3 / 2, e: -f / 2 - s5 / 2, g: 2}]