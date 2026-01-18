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
def test_linear_coeffs():
    from sympy.solvers.solveset import linear_coeffs
    assert linear_coeffs(0, x) == [0, 0]
    assert all((i is S.Zero for i in linear_coeffs(0, x)))
    assert linear_coeffs(x + 2 * y + 3, x, y) == [1, 2, 3]
    assert linear_coeffs(x + 2 * y + 3, y, x) == [2, 1, 3]
    assert linear_coeffs(x + 2 * x ** 2 + 3, x, x ** 2) == [1, 2, 3]
    raises(ValueError, lambda: linear_coeffs(x + 2 * x ** 2 + x ** 3, x, x ** 2))
    raises(ValueError, lambda: linear_coeffs(1 / x * (x - 1) + 1 / x, x))
    raises(ValueError, lambda: linear_coeffs(x, x, x))
    assert linear_coeffs(a * (x + y), x, y) == [a, a, 0]
    assert linear_coeffs(1.0, x, y) == [0, 0, 1.0]
    assert linear_coeffs(Eq(x, x + y), x, y, dict=True) == {y: -1}
    assert linear_coeffs(0, x, y, dict=True) == {}