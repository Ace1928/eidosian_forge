from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import I, Rational, pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.core.function import ArgumentIndexError
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises
def test_betainc_regularized():
    a, b, x1, x2 = symbols('a b x1 x2')
    assert unchanged(betainc_regularized, a, b, x1, x2)
    assert unchanged(betainc_regularized, a, b, 0, x1)
    assert betainc_regularized(3, 5, 0, -1).is_real == True
    assert betainc_regularized(3, 5, 0, x2).is_real is None
    assert conjugate(betainc_regularized(3 * I, 1, 2 + I, 1 + 2 * I)) == betainc_regularized(-3 * I, 1, 2 - I, 1 - 2 * I)
    assert betainc_regularized(a, b, 0, 1).rewrite(Integral) == 1
    assert betainc_regularized(1, 2, x1, x2).rewrite(hyper) == 2 * x2 * hyper((1, -1), (2,), x2) - 2 * x1 * hyper((1, -1), (2,), x1)
    assert betainc_regularized(4, 1, 5, 5).evalf() == 0