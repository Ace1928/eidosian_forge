from math import isclose
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.functions.special.error_functions import (erf, erfc,
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols
from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm
from sympy.solvers import solve
from sympy.solvers.solveset import (
from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
def test_issue_14642():
    x = Symbol('x')
    n1 = 0.5 * x ** 3 + x ** 2 + 0.5 + I
    solution = solveset(n1, x)
    assert abs(solution.args[0] - (-2.28267560928153 - 0.312325580497716 * I)) <= 1e-09
    assert abs(solution.args[1] - (-0.297354141679308 + 1.01904778618762 * I)) <= 1e-09
    assert abs(solution.args[2] - (0.580029750960839 - 0.706722205689907 * I)) <= 1e-09
    n1 = S.Half * x ** 3 + x ** 2 + S.Half + I
    res = FiniteSet(-((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * cos(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 3 - S(2) / 3 - 4 * cos(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / (3 * ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6)) + I * (-((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * sin(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 3 + 4 * sin(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / (3 * ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6))), -S(2) / 3 - sqrt(3) * ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * sin(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6 - 4 * re(1 / ((-S(1) / 2 - sqrt(3) * I / 2) * (S(43) / 2 + 27 * I + sqrt(-256 + (43 + 54 * I) ** 2) / 2) ** (S(1) / 3))) / 3 + ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * cos(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6 + I * (-4 * im(1 / ((-S(1) / 2 - sqrt(3) * I / 2) * (S(43) / 2 + 27 * I + sqrt(-256 + (43 + 54 * I) ** 2) / 2) ** (S(1) / 3))) / 3 + ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * sin(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6 + sqrt(3) * ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * cos(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6), -S(2) / 3 - 4 * re(1 / ((-S(1) / 2 + sqrt(3) * I / 2) * (S(43) / 2 + 27 * I + sqrt(-256 + (43 + 54 * I) ** 2) / 2) ** (S(1) / 3))) / 3 + sqrt(3) * ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * sin(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6 + ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * cos(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6 + I * (-sqrt(3) * ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * cos(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6 + ((3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2) ** 2 + (27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) ** 2) ** (S(1) / 6) * sin(atan((27 + 3 * sqrt(3) * 31985 ** (S(1) / 4) * cos(atan(S(172) / 49) / 2) / 2) / (3 * sqrt(3) * 31985 ** (S(1) / 4) * sin(atan(S(172) / 49) / 2) / 2 + S(43) / 2)) / 3) / 6 - 4 * im(1 / ((-S(1) / 2 + sqrt(3) * I / 2) * (S(43) / 2 + 27 * I + sqrt(-256 + (43 + 54 * I) ** 2) / 2) ** (S(1) / 3))) / 3))
    assert solveset(n1, x) == res