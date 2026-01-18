from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.numbers import (Float, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.logic.boolalg import (And, Not)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.stats import (DiscreteMarkovChain, P, TransitionMatrixOf, E,
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import RandomIndexedSymbol
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.testing.pytest import (raises, skip, ignore_warnings,
from sympy.external import import_module
from sympy.stats.frv_types import BernoulliDistribution
from sympy.stats.drv_types import PoissonDistribution
from sympy.stats.crv_types import NormalDistribution, GammaDistribution
from sympy.core.symbol import Str
def test_GammaProcess_numeric():
    t, d, x, y = symbols('t d x y', positive=True)
    X = GammaProcess('X', 1, 2)
    assert X.state_space == Interval(0, oo)
    assert X.index_set == Interval(0, oo)
    assert X.lamda == 1
    assert X.gamma == 2
    raises(ValueError, lambda: GammaProcess('X', -1, 2))
    raises(ValueError, lambda: GammaProcess('X', 0, -2))
    raises(ValueError, lambda: GammaProcess('X', -1, -2))
    assert P((X(t) > 4) & (X(d) > 3) & (X(x) > 2) & (X(y) > 1), Contains(t, Interval.Lopen(0, 1)) & Contains(d, Interval.Lopen(1, 2)) & Contains(x, Interval.Lopen(2, 3)) & Contains(y, Interval.Lopen(3, 4))).simplify() == 120 * exp(-10)
    assert P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) & Contains(d, Interval.Lopen(7, 8))).simplify() == -4 * exp(-3) + 472 * exp(-8) / 3 + 1
    assert P((X(t) > 2) | (X(t) < 4), Contains(t, Interval.Ropen(1, 4))).simplify() == -643 * exp(-4) / 15 + 109 * exp(-2) / 15 + 1
    assert E(X(t)) == 2 * t
    assert E(X(2) + x * E(X(5))) == 10 * x + 4