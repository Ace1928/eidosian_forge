from sympy.concrete.summations import Sum
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.beta_functions import beta
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.matrices import Matrix
from sympy.stats import (DiscreteUniform, Die, Bernoulli, Coin, Binomial, BetaBinomial,
from sympy.stats.frv_types import DieDistribution, BinomialDistribution, \
from sympy.stats.rv import Density
from sympy.testing.pytest import raises
def test_DieDistribution():
    from sympy.abc import x
    X = DieDistribution(6)
    assert X.pmf(S.Half) is S.Zero
    assert X.pmf(x).subs({x: 1}).doit() == Rational(1, 6)
    assert X.pmf(x).subs({x: 7}).doit() == 0
    assert X.pmf(x).subs({x: -1}).doit() == 0
    assert X.pmf(x).subs({x: Rational(1, 3)}).doit() == 0
    raises(ValueError, lambda: X.pmf(Matrix([0, 0])))
    raises(ValueError, lambda: X.pmf(x ** 2 - 1))