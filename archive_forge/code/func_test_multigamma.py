from sympy.core.function import expand_func, Subs
from sympy.core import EulerGamma
from sympy.core.numbers import (I, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, atan)
from sympy.functions.special.error_functions import (Ei, erf, erfc)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, lowergamma, multigamma, polygamma, trigamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
from sympy.core.random import (test_derivative_numerically as td,
def test_multigamma():
    from sympy.concrete.products import Product
    p = Symbol('p')
    _k = Dummy('_k')
    assert multigamma(x, p).dummy_eq(pi ** (p * (p - 1) / 4) * Product(gamma(x + (1 - _k) / 2), (_k, 1, p)))
    assert conjugate(multigamma(x, p)).dummy_eq(pi ** ((conjugate(p) - 1) * conjugate(p) / 4) * Product(gamma(conjugate(x) + (1 - conjugate(_k)) / 2), (_k, 1, p)))
    assert conjugate(multigamma(x, 1)) == gamma(conjugate(x))
    p = Symbol('p', positive=True)
    assert conjugate(multigamma(x, p)).dummy_eq(pi ** ((p - 1) * p / 4) * Product(gamma(conjugate(x) + (1 - conjugate(_k)) / 2), (_k, 1, p)))
    assert multigamma(nan, 1) is nan
    assert multigamma(oo, 1).doit() is oo
    assert multigamma(1, 1) == 1
    assert multigamma(2, 1) == 1
    assert multigamma(3, 1) == 2
    assert multigamma(102, 1) == factorial(101)
    assert multigamma(S.Half, 1) == sqrt(pi)
    assert multigamma(1, 2) == pi
    assert multigamma(2, 2) == pi / 2
    assert multigamma(1, 3) is zoo
    assert multigamma(2, 3) == pi ** 2 / 2
    assert multigamma(3, 3) == 3 * pi ** 2 / 2
    assert multigamma(x, 1).diff(x) == gamma(x) * polygamma(0, x)
    assert multigamma(x, 2).diff(x) == sqrt(pi) * gamma(x) * gamma(x - S.Half) * polygamma(0, x) + sqrt(pi) * gamma(x) * gamma(x - S.Half) * polygamma(0, x - S.Half)
    assert multigamma(x - 1, 1).expand(func=True) == gamma(x) / (x - 1)
    assert multigamma(x + 2, 1).expand(func=True, mul=False) == x * (x + 1) * gamma(x)
    assert multigamma(x - 1, 2).expand(func=True) == sqrt(pi) * gamma(x) * gamma(x + S.Half) / (x ** 3 - 3 * x ** 2 + x * Rational(11, 4) - Rational(3, 4))
    assert multigamma(x - 1, 3).expand(func=True) == pi ** Rational(3, 2) * gamma(x) ** 2 * gamma(x + S.Half) / (x ** 5 - 6 * x ** 4 + 55 * x ** 3 / 4 - 15 * x ** 2 + x * Rational(31, 4) - Rational(3, 2))
    assert multigamma(n, 1).rewrite(factorial) == factorial(n - 1)
    assert multigamma(n, 2).rewrite(factorial) == sqrt(pi) * factorial(n - Rational(3, 2)) * factorial(n - 1)
    assert multigamma(n, 3).rewrite(factorial) == pi ** Rational(3, 2) * factorial(n - 2) * factorial(n - Rational(3, 2)) * factorial(n - 1)
    assert multigamma(Rational(-1, 2), 3, evaluate=False).is_real == False
    assert multigamma(S.Half, 3, evaluate=False).is_real == False
    assert multigamma(0, 1, evaluate=False).is_real == False
    assert multigamma(1, 3, evaluate=False).is_real == False
    assert multigamma(-1.0, 3, evaluate=False).is_real == False
    assert multigamma(0.7, 3, evaluate=False).is_real == True
    assert multigamma(3, 3, evaluate=False).is_real == True