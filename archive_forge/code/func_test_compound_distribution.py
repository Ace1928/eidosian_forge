from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Ne)
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.stats import (Poisson, Beta, Exponential, P,
from sympy.stats.crv_types import Normal
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
from sympy.stats.joint_rv import MarginalDistribution
from sympy.stats.rv import pspace, density
from sympy.testing.pytest import ignore_warnings
def test_compound_distribution():
    Y = Poisson('Y', 1)
    Z = Poisson('Z', Y)
    assert isinstance(pspace(Z), CompoundPSpace)
    assert isinstance(pspace(Z).distribution, CompoundDistribution)
    assert Z.pspace.distribution.pdf(1).doit() == exp(-2) * exp(exp(-1))