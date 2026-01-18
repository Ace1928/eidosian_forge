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
def test_exclude():
    R, C, Ri, Vout, V1, Vminus, Vplus, s = symbols('R, C, Ri, Vout, V1, Vminus, Vplus, s')
    Rf = symbols('Rf', positive=True)
    eqs = [C * V1 * s + Vplus * (-2 * C * s - 1 / R), Vminus * (-1 / Ri - 1 / Rf) + Vout / Rf, C * Vplus * s + V1 * (-C * s - 1 / R) + Vout / R, -Vminus + Vplus]
    assert solve(eqs, exclude=s * C * R) == [{Rf: Ri * (C * R * s + 1) ** 2 / (C * R * s), Vminus: Vplus, V1: 2 * Vplus + Vplus / (C * R * s), Vout: C * R * Vplus * s + 3 * Vplus + Vplus / (C * R * s)}, {Vplus: 0, Vminus: 0, V1: 0, Vout: 0}]
    assert solve(eqs, exclude=[Vplus, s, C]) in [[{Vminus: Vplus, V1: Vout / 2 + Vplus / 2 + sqrt((Vout - 5 * Vplus) * (Vout - Vplus)) / 2, R: (Vout - 3 * Vplus - sqrt(Vout ** 2 - 6 * Vout * Vplus + 5 * Vplus ** 2)) / (2 * C * Vplus * s), Rf: Ri * (Vout - Vplus) / Vplus}, {Vminus: Vplus, V1: Vout / 2 + Vplus / 2 - sqrt((Vout - 5 * Vplus) * (Vout - Vplus)) / 2, R: (Vout - 3 * Vplus + sqrt(Vout ** 2 - 6 * Vout * Vplus + 5 * Vplus ** 2)) / (2 * C * Vplus * s), Rf: Ri * (Vout - Vplus) / Vplus}], [{Vminus: Vplus, Vout: (V1 ** 2 - V1 * Vplus - Vplus ** 2) / (V1 - 2 * Vplus), Rf: Ri * (V1 - Vplus) ** 2 / (Vplus * (V1 - 2 * Vplus)), R: Vplus / (C * s * (V1 - 2 * Vplus))}]]