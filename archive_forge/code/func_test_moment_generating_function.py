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
def test_moment_generating_function():
    X = Normal('X', 0, 1)
    Y = DiscreteUniform('Y', [1, 2, 7])
    Z = Poisson('Z', 2)
    t = symbols('_t')
    P = Lambda(t, exp(t ** 2 / 2))
    Q = Lambda(t, exp(7 * t) / 3 + exp(2 * t) / 3 + exp(t) / 3)
    R = Lambda(t, exp(2 * exp(t) - 2))
    assert moment_generating_function(X).dummy_eq(P)
    assert moment_generating_function(Y).dummy_eq(Q)
    assert moment_generating_function(Z).dummy_eq(R)