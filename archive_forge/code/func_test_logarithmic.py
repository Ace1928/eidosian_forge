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
def test_logarithmic():
    assert solveset_real(log(x - 3) + log(x + 3), x) == FiniteSet(-sqrt(10), sqrt(10))
    assert solveset_real(log(x + 1) - log(2 * x - 1), x) == FiniteSet(2)
    assert solveset_real(log(x + 3) + log(1 + 3 / x) - 3, x) == FiniteSet(-3 + sqrt(-12 + exp(3)) * exp(Rational(3, 2)) / 2 + exp(3) / 2, -sqrt(-12 + exp(3)) * exp(Rational(3, 2)) / 2 - 3 + exp(3) / 2)
    eq = z - log(x) + log(y / (x * (-1 + y ** 2 / x ** 2)))
    assert solveset_real(eq, x) == Intersection(S.Reals, FiniteSet(-sqrt(y ** 2 - y * exp(z)), sqrt(y ** 2 - y * exp(z)))) - Intersection(S.Reals, FiniteSet(-sqrt(y ** 2), sqrt(y ** 2)))
    assert solveset_real(log(3 * x) - log(-x + 1) - log(4 * x + 1), x) == FiniteSet(Rational(-1, 2), S.Half)
    assert solveset(log(x ** y) - y * log(x), x, S.Reals) == S.Reals