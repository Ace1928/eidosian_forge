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
def test_issue_4793():
    assert solve(1 / x) == []
    assert solve(x * (1 - 5 / x)) == [5]
    assert solve(x + sqrt(x) - 2) == [1]
    assert solve(-(1 + x) / (2 + x) ** 2 + 1 / (2 + x)) == []
    assert solve(-x ** 2 - 2 * x + (x + 1) ** 2 - 1) == []
    assert solve((x / (x + 1) + 3) ** (-2)) == []
    assert solve(x / sqrt(x ** 2 + 1), x) == [0]
    assert solve(exp(x) - y, x) == [log(y)]
    assert solve(exp(x)) == []
    assert solve(x ** 2 + x + sin(y) ** 2 + cos(y) ** 2 - 1, x) in [[0, -1], [-1, 0]]
    eq = 4 * 3 ** (5 * x + 2) - 7
    ans = solve(eq, x)
    assert len(ans) == 5 and all((eq.subs(x, a).n(chop=True) == 0 for a in ans))
    assert solve(log(x ** 2) - y ** 2 / exp(x), x, y, set=True) == ([x, y], {(x, sqrt(exp(x) * log(x ** 2))), (x, -sqrt(exp(x) * log(x ** 2)))})
    assert solve(x ** 2 * z ** 2 - z ** 2 * y ** 2) == [{x: -y}, {x: y}, {z: 0}]
    assert solve((x - 1) / (1 + 1 / (x - 1))) == []
    assert solve(x ** (y * z) - x, x) == [1]
    raises(NotImplementedError, lambda: solve(log(x) - exp(x), x))
    raises(NotImplementedError, lambda: solve(2 ** x - exp(x) - 3))