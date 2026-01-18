from math import prod
from sympy.concrete.expr_with_intlimits import ReorderError
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import (Sum, summation, telescopic,
from sympy.core.function import (Derivative, Function)
from sympy.core import (Catalan, EulerGamma)
from sympy.core.facts import InconsistentAssumptions
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sinh, tanh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Or
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices import (Matrix, SparseMatrix,
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.abc import a, b, c, d, k, m, x, y, z
def test_evalf_fast_series():
    assert NS(Sum(fac(2 * n + 1) / fac(n) ** 2 / 2 ** (3 * n + 1), (n, 0, oo)), 100) == NS(sqrt(2), 100)
    estr = NS(E, 100)
    assert NS(Sum(1 / fac(n), (n, 0, oo)), 100) == estr
    assert NS(1 / Sum((1 - 2 * n) / fac(2 * n), (n, 0, oo)), 100) == estr
    assert NS(Sum((2 * n + 1) / fac(2 * n), (n, 0, oo)), 100) == estr
    assert NS(Sum((4 * n + 3) / 2 ** (2 * n + 1) / fac(2 * n + 1), (n, 0, oo)) ** 2, 100) == estr
    pistr = NS(pi, 100)
    assert NS(9801 / sqrt(8) / Sum(fac(4 * n) * (1103 + 26390 * n) / fac(n) ** 4 / 396 ** (4 * n), (n, 0, oo)), 100) == pistr
    assert NS(1 / Sum(binomial(2 * n, n) ** 3 * (42 * n + 5) / 2 ** (12 * n + 4), (n, 0, oo)), 100) == pistr
    assert NS(16 * Sum((-1) ** n / (2 * n + 1) / 5 ** (2 * n + 1), (n, 0, oo)) - 4 * Sum((-1) ** n / (2 * n + 1) / 239 ** (2 * n + 1), (n, 0, oo)), 100) == pistr
    astr = NS(zeta(3), 100)
    P = 126392 * n ** 5 + 412708 * n ** 4 + 531578 * n ** 3 + 336367 * n ** 2 + 104000 * n + 12463
    assert NS(Sum((-1) ** n * P / 24 * (fac(2 * n + 1) * fac(2 * n) * fac(n)) ** 3 / fac(3 * n + 2) / fac(4 * n + 3) ** 3, (n, 0, oo)), 100) == astr
    assert NS(Sum((-1) ** n * (205 * n ** 2 + 250 * n + 77) / 64 * fac(n) ** 10 / fac(2 * n + 1) ** 5, (n, 0, oo)), 100) == astr