from sympy.concrete.summations import Sum
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.sets import Interval
from sympy.stats import (Normal, P, E, density, Gamma, Poisson, Rayleigh,
from sympy.stats.compound_rv import CompoundDistribution, CompoundPSpace
from sympy.stats.crv_types import NormalDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.frv_types import BernoulliDistribution
from sympy.testing.pytest import raises, ignore_warnings
from sympy.stats.joint_rv_types import MultivariateNormalDistribution
from sympy.abc import x
def test_Compound_Distribution():
    X = Normal('X', 2, 4)
    N = NormalDistribution(X, 4)
    C = CompoundDistribution(N)
    assert C.is_Continuous
    assert C.set == Interval(-oo, oo)
    assert C.pdf(x, evaluate=True).simplify() == exp(-x ** 2 / 64 + x / 16 - S(1) / 16) / (8 * sqrt(pi))
    assert not isinstance(CompoundDistribution(NormalDistribution(2, 3)), CompoundDistribution)
    M = MultivariateNormalDistribution([1, 2], [[2, 1], [1, 2]])
    raises(NotImplementedError, lambda: CompoundDistribution(M))
    X = Beta('X', 2, 4)
    B = BernoulliDistribution(X, 1, 0)
    C = CompoundDistribution(B)
    assert C.is_Finite
    assert C.set == {0, 1}
    y = symbols('y', negative=False, integer=True)
    assert C.pdf(y, evaluate=True) == Piecewise((S(1) / (30 * beta(2, 4)), Eq(y, 0)), (S(1) / (60 * beta(2, 4)), Eq(y, 1)), (0, True))
    k, t, z = symbols('k t z', positive=True, real=True)
    G = Gamma('G', k, t)
    X = PoissonDistribution(G)
    C = CompoundDistribution(X)
    assert C.is_Discrete
    assert C.set == S.Naturals0
    assert C.pdf(z, evaluate=True).simplify() == t ** z * (t + 1) ** (-k - z) * gamma(k + z) / (gamma(k) * gamma(z + 1))