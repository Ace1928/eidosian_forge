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
def test_issue_14129():
    x = Symbol('x', zero=False)
    assert Sum(k * x ** k, (k, 0, n - 1)).doit() == Piecewise((n ** 2 / 2 - n / 2, Eq(x, 1)), ((n * x * x ** n - n * x ** n - x * x ** n + x) / (x - 1) ** 2, True))
    assert Sum(x ** k, (k, 0, n - 1)).doit() == Piecewise((n, Eq(x, 1)), ((-x ** n + 1) / (-x + 1), True))
    assert Sum(k * (x / y + x) ** k, (k, 0, n - 1)).doit() == Piecewise((n * (n - 1) / 2, Eq(x, y / (y + 1))), (x * (y + 1) * (n * x * y * (x + x / y) ** (n - 1) + n * x * (x + x / y) ** (n - 1) - n * y * (x + x / y) ** (n - 1) - x * y * (x + x / y) ** (n - 1) - x * (x + x / y) ** (n - 1) + y) / (x * y + x - y) ** 2, True))