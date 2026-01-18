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
def test_compound_pspace():
    X = Normal('X', 2, 4)
    Y = Normal('Y', 3, 6)
    assert not isinstance(Y.pspace, CompoundPSpace)
    N = NormalDistribution(1, 2)
    D = PoissonDistribution(3)
    B = BernoulliDistribution(0.2, 1, 0)
    pspace1 = CompoundPSpace('N', N)
    pspace2 = CompoundPSpace('D', D)
    pspace3 = CompoundPSpace('B', B)
    assert not isinstance(pspace1, CompoundPSpace)
    assert not isinstance(pspace2, CompoundPSpace)
    assert not isinstance(pspace3, CompoundPSpace)
    M = MultivariateNormalDistribution([1, 2], [[2, 1], [1, 2]])
    raises(ValueError, lambda: CompoundPSpace('M', M))
    Y = Normal('Y', X, 6)
    assert isinstance(Y.pspace, CompoundPSpace)
    assert Y.pspace.distribution == CompoundDistribution(NormalDistribution(X, 6))
    assert Y.pspace.domain.set == Interval(-oo, oo)