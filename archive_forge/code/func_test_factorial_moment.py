from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.numbers import (Rational, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (FallingFactorial, binomial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import DiracDelta
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.tensor.indexed import Indexed
from sympy.stats import (Die, Normal, Exponential, FiniteRV, P, E, H, variance,
from sympy.stats.rv import (IndependentProductPSpace, rs_swap, Density, NamedArgsMixin,
from sympy.testing.pytest import raises, skip, XFAIL, warns_deprecated_sympy
from sympy.external import import_module
from sympy.core.numbers import comp
from sympy.stats.frv_types import BernoulliDistribution
from sympy.core.symbol import Dummy
from sympy.functions.elementary.piecewise import Piecewise
def test_factorial_moment():
    X = Poisson('X', 2)
    Y = Binomial('Y', 2, S.Half)
    Z = Hypergeometric('Z', 4, 2, 2)
    assert factorial_moment(X, 2) == 4
    assert factorial_moment(Y, 2) == S.Half
    assert factorial_moment(Z, 2) == Rational(1, 3)
    x, y, z, l = symbols('x y z l')
    Y = Binomial('Y', 2, y)
    Z = Hypergeometric('Z', 10, 2, 3)
    assert factorial_moment(Y, l) == y ** 2 * FallingFactorial(2, l) + 2 * y * (1 - y) * FallingFactorial(1, l) + (1 - y) ** 2 * FallingFactorial(0, l)
    assert factorial_moment(Z, l) == 7 * FallingFactorial(0, l) / 15 + 7 * FallingFactorial(1, l) / 15 + FallingFactorial(2, l) / 15