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
def test_conditionset():
    assert solveset(Eq(sin(x) ** 2 + cos(x) ** 2, 1), x, domain=S.Reals) is S.Reals
    assert solveset(Eq(x ** 2 + x * sin(x), 1), x, domain=S.Reals).dummy_eq(ConditionSet(x, Eq(x ** 2 + x * sin(x) - 1, 0), S.Reals))
    assert dumeq(solveset(Eq(-I * (exp(I * x) - exp(-I * x)) / 2, 1), x), imageset(Lambda(n, 2 * n * pi + pi / 2), S.Integers))
    assert solveset(x + sin(x) > 1, x, domain=S.Reals).dummy_eq(ConditionSet(x, x + sin(x) > 1, S.Reals))
    assert solveset(Eq(sin(Abs(x)), x), x, domain=S.Reals).dummy_eq(ConditionSet(x, Eq(-x + sin(Abs(x)), 0), S.Reals))
    assert solveset(y ** x - z, x, S.Reals).dummy_eq(ConditionSet(x, Eq(y ** x - z, 0), S.Reals))