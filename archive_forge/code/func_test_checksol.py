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
def test_checksol():
    x, y, r, t = symbols('x, y, r, t')
    eq = r - x ** 2 - y ** 2
    dict_var_soln = {y: -sqrt(r) / sqrt(tan(t) ** 2 + 1), x: -sqrt(r) * tan(t) / sqrt(tan(t) ** 2 + 1)}
    assert checksol(eq, dict_var_soln) == True
    assert checksol(Eq(x, False), {x: False}) is True
    assert checksol(Ne(x, False), {x: False}) is False
    assert checksol(Eq(x < 1, True), {x: 0}) is True
    assert checksol(Eq(x < 1, True), {x: 1}) is False
    assert checksol(Eq(x < 1, False), {x: 1}) is True
    assert checksol(Eq(x < 1, False), {x: 0}) is False
    assert checksol(Eq(x + 1, x ** 2 + 1), {x: 1}) is True
    assert checksol([x - 1, x ** 2 - 1], x, 1) is True
    assert checksol([x - 1, x ** 2 - 2], x, 1) is False
    assert checksol(Poly(x ** 2 - 1), x, 1) is True
    assert checksol(0, {}) is True
    assert checksol([1e-10, x - 2], x, 2) is False
    assert checksol([0.5, 0, x], x, 0) is False
    assert checksol(y, x, 2) is False
    assert checksol(x + 1e-10, x, 0, numerical=True) is True
    assert checksol(x + 1e-10, x, 0, numerical=False) is False
    assert checksol(exp(92 * x), {x: log(sqrt(2) / 2)}) is False
    assert checksol(exp(92 * x), {x: log(sqrt(2) / 2) + I * pi}) is False
    assert checksol(1 / x ** 5, x, 1000) is False
    raises(ValueError, lambda: checksol(x, 1))
    raises(ValueError, lambda: checksol([], x, 1))