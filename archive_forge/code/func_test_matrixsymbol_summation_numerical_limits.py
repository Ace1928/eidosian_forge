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
def test_matrixsymbol_summation_numerical_limits():
    A = MatrixSymbol('A', 3, 3)
    n = Symbol('n', integer=True)
    assert Sum(A ** n, (n, 0, 2)).doit() == Identity(3) + A + A ** 2
    assert Sum(A, (n, 0, 2)).doit() == 3 * A
    assert Sum(n * A, (n, 0, 2)).doit() == 3 * A
    B = Matrix([[0, n, 0], [-1, 0, 0], [0, 0, 2]])
    ans = Matrix([[0, 6, 0], [-4, 0, 0], [0, 0, 8]]) + 4 * A
    assert Sum(A + B, (n, 0, 3)).doit() == ans
    ans = A * Matrix([[0, 6, 0], [-4, 0, 0], [0, 0, 8]])
    assert Sum(A * B, (n, 0, 3)).doit() == ans
    ans = A ** 2 * Matrix([[-2, 0, 0], [0, -2, 0], [0, 0, 4]]) + A ** 3 * Matrix([[0, -9, 0], [3, 0, 0], [0, 0, 8]]) + A * Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 2]])
    assert Sum(A ** n * B ** n, (n, 1, 3)).doit() == ans