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
def test_issue_21022():
    from sympy.core.sympify import sympify
    eqs = ['k-16', 'p-8', 'y*y+z*z-x*x', 'd - x + p', 'd*d+k*k-y*y', 'z*z-p*p-k*k', 'abc-efg']
    efg = Symbol('efg')
    eqs = [sympify(x) for x in eqs]
    syb = list(ordered(set.union(*[x.free_symbols for x in eqs])))
    res = nonlinsolve(eqs, syb)
    ans = FiniteSet((efg, 32, efg, 16, 8, 40, -16 * sqrt(5), -8 * sqrt(5)), (efg, 32, efg, 16, 8, 40, -16 * sqrt(5), 8 * sqrt(5)), (efg, 32, efg, 16, 8, 40, 16 * sqrt(5), -8 * sqrt(5)), (efg, 32, efg, 16, 8, 40, 16 * sqrt(5), 8 * sqrt(5)))
    assert len(res) == len(ans) == 4
    assert res == ans
    for result in res.args:
        assert len(result) == 8