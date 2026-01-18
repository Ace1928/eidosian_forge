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
def test_factor_expand_subs():
    assert Sum(4 * x, (x, 1, y)).factor() == 4 * Sum(x, (x, 1, y))
    assert Sum(x * a, (x, 1, y)).factor() == a * Sum(x, (x, 1, y))
    assert Sum(4 * x * a, (x, 1, y)).factor() == 4 * a * Sum(x, (x, 1, y))
    assert Sum(4 * x * y, (x, 1, y)).factor() == 4 * y * Sum(x, (x, 1, y))
    _x = Symbol('x', zero=False)
    assert Sum(x + 1, (x, 1, y)).expand() == Sum(x, (x, 1, y)) + Sum(1, (x, 1, y))
    assert Sum(x + a * x ** 2, (x, 1, y)).expand() == Sum(x, (x, 1, y)) + Sum(a * x ** 2, (x, 1, y))
    assert Sum(_x ** (n + 1) * (n + 1), (n, -1, oo)).expand() == Sum(n * _x * _x ** n + _x * _x ** n, (n, -1, oo))
    assert Sum(x ** (n + 1) * (n + 1), (n, -1, oo)).expand(power_exp=False) == Sum(n * x ** (n + 1) + x ** (n + 1), (n, -1, oo))
    assert Sum(x ** (n + 1) * (n + 1), (n, -1, oo)).expand(force=True) == Sum(x * x ** n, (n, -1, oo)) + Sum(n * x * x ** n, (n, -1, oo))
    assert Sum(a * n + a * n ** 2, (n, 0, 4)).expand() == Sum(a * n, (n, 0, 4)) + Sum(a * n ** 2, (n, 0, 4))
    assert Sum(_x ** a * _x ** n, (x, 0, 3)) == Sum(_x ** (a + n), (x, 0, 3)).expand(power_exp=True)
    _a, _n = symbols('a n', positive=True)
    assert Sum(x ** (_a + _n), (x, 0, 3)).expand(power_exp=True) == Sum(x ** _a * x ** _n, (x, 0, 3))
    assert Sum(x ** (_a - _n), (x, 0, 3)).expand(power_exp=True) == Sum(x ** (_a - _n), (x, 0, 3)).expand(power_exp=False)
    assert Sum(1 / (1 + a * x ** 2), (x, 0, 3)).subs([(a, 3)]) == Sum(1 / (1 + 3 * x ** 2), (x, 0, 3))
    assert Sum(x * y, (x, 0, y), (y, 0, x)).subs([(x, 3)]) == Sum(x * y, (x, 0, y), (y, 0, 3))
    assert Sum(x, (x, 1, 10)).subs([(x, y - 2)]) == Sum(x, (x, 1, 10))
    assert Sum(1 / x, (x, 1, 10)).subs([(x, (3 + n) ** 3)]) == Sum(1 / x, (x, 1, 10))
    assert Sum(1 / x, (x, 1, 10)).subs([(x, 3 * x - 2)]) == Sum(1 / x, (x, 1, 10))