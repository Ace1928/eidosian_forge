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
def test_bernoulli_CompoundDist():
    X = Beta('X', 1, 2)
    Y = Bernoulli('Y', X)
    assert density(Y).dict == {0: S(2) / 3, 1: S(1) / 3}
    assert E(Y) == P(Eq(Y, 1)) == S(1) / 3
    assert variance(Y) == S(2) / 9
    assert cdf(Y) == {0: S(2) / 3, 1: 1}
    a = Bernoulli('a', S(1) / 2)
    b = Bernoulli('b', a)
    assert density(b).dict == {0: S(1) / 2, 1: S(1) / 2}
    assert P(b > 0.5) == S(1) / 2
    X = Uniform('X', 0, 1)
    Y = Bernoulli('Y', X)
    assert E(Y) == S(1) / 2
    assert P(Eq(Y, 1)) == E(Y)