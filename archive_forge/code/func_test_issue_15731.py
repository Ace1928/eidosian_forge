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
def test_issue_15731():
    assert solve(Eq((x ** 2 - 7 * x + 11) ** (x ** 2 - 13 * x + 42), 1)) == [2, 3, 4, 5, 6, 7]
    assert solve(x ** (x + 4) - 4) == [-2]
    assert solve((-x) ** (-x + 4) - 4) == [2]
    assert solve((x ** 2 - 6) ** (x ** 2 - 2) - 4) == [-2, 2]
    assert solve((x ** 2 - 2 * x - 1) ** (x ** 2 - 3) - 1 / (1 - 2 * sqrt(2))) == [sqrt(2)]
    assert solve(x ** (x + S.Half) - 4 * sqrt(2)) == [S(2)]
    assert solve((x ** 2 + 1) ** x - 25) == [2]
    assert solve(x ** (2 / x) - 2) == [2, 4]
    assert solve((x / 2) ** (2 / x) - sqrt(2)) == [4, 8]
    assert solve(x ** (x + S.Half) - Rational(9, 4)) == [Rational(3, 2)]
    assert solve((-sqrt(sqrt(2))) ** x - 2) == [4, log(2) / (log(2 ** Rational(1, 4)) + I * pi)]
    assert solve(sqrt(2) ** x - sqrt(sqrt(2))) == [S.Half]
    assert solve((-sqrt(2)) ** x + 2 * sqrt(2)) == [3, (3 * log(2) ** 2 + 4 * pi ** 2 - 4 * I * pi * log(2)) / (log(2) ** 2 + 4 * pi ** 2)]
    assert solve(sqrt(2) ** x - 2 * sqrt(2)) == [3]
    assert solve(I ** x + 1) == [2]
    assert solve((1 + I) ** x - 2 * I) == [2]
    assert solve((sqrt(2) + sqrt(3)) ** x - (2 * sqrt(6) + 5) ** Rational(1, 3)) == [Rational(2, 3)]
    b = Symbol('b')
    assert solve(b ** x - b ** 2, x) == [2]
    assert solve(b ** x - 1 / b, x) == [-1]
    assert solve(b ** x - b, x) == [1]
    b = Symbol('b', positive=True)
    assert solve(b ** x - b ** 2, x) == [2]
    assert solve(b ** x - 1 / b, x) == [-1]