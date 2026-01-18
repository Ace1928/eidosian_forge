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
def test_RandomSymbol():
    X = Normal('x', 0, 1)
    Y = Normal('x', 0, 2)
    assert X.symbol == Y.symbol
    assert X != Y
    assert X.name == X.symbol.name
    X = Normal('lambda', 0, 1)
    X = Normal('Lambda', 0, 1)