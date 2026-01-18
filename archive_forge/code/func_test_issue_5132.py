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
def test_issue_5132():
    r, t = symbols('r,t')
    assert set(solve([r - x ** 2 - y ** 2, tan(t) - y / x], [x, y])) == {(-sqrt(r * cos(t) ** 2), -1 * sqrt(r * cos(t) ** 2) * tan(t)), (sqrt(r * cos(t) ** 2), sqrt(r * cos(t) ** 2) * tan(t))}
    assert solve([exp(x) - sin(y), 1 / y - 3], [x, y]) == [(log(sin(Rational(1, 3))), Rational(1, 3))]
    assert solve([exp(x) - sin(y), 1 / exp(y) - 3], [x, y]) == [(log(-sin(log(3))), -log(3))]
    assert set(solve([exp(x) - sin(y), y ** 2 - 4], [x, y])) == {(log(-sin(2)), -S(2)), (log(sin(2)), S(2))}
    eqs = [exp(x) ** 2 - sin(y) + z ** 2, 1 / exp(y) - 3]
    assert solve(eqs, set=True) == ([y, z], {(-log(3), sqrt(-exp(2 * x) - sin(log(3)))), (-log(3), -sqrt(-exp(2 * x) - sin(log(3))))})
    assert solve(eqs, x, z, set=True) == ([x, z], {(x, sqrt(-exp(2 * x) + sin(y))), (x, -sqrt(-exp(2 * x) + sin(y)))})
    assert set(solve(eqs, x, y)) == {(log(-sqrt(-z ** 2 - sin(log(3)))), -log(3)), (log(-z ** 2 - sin(log(3))) / 2, -log(3))}
    assert set(solve(eqs, y, z)) == {(-log(3), -sqrt(-exp(2 * x) - sin(log(3)))), (-log(3), sqrt(-exp(2 * x) - sin(log(3))))}
    eqs = [exp(x) ** 2 - sin(y) + z, 1 / exp(y) - 3]
    assert solve(eqs, set=True) == ([y, z], {(-log(3), -exp(2 * x) - sin(log(3)))})
    assert solve(eqs, x, z, set=True) == ([x, z], {(x, -exp(2 * x) + sin(y))})
    assert set(solve(eqs, x, y)) == {(log(-sqrt(-z - sin(log(3)))), -log(3)), (log(-z - sin(log(3))) / 2, -log(3))}
    assert solve(eqs, z, y) == [(-exp(2 * x) - sin(log(3)), -log(3))]
    assert solve((sqrt(x ** 2 + y ** 2) - sqrt(10), x + y - 4), set=True) == ([x, y], {(S.One, S(3)), (S(3), S.One)})
    assert set(solve((sqrt(x ** 2 + y ** 2) - sqrt(10), x + y - 4), x, y)) == {(S.One, S(3)), (S(3), S.One)}