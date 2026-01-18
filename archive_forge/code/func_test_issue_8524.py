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
def test_issue_8524():
    x = Symbol('x', positive=True)
    y = Symbol('y', negative=True)
    z = Symbol('z', positive=False)
    p = Symbol('p', negative=False)
    q = Symbol('q', integer=True)
    r = Symbol('r', integer=False)
    e = Symbol('e', even=True, negative=True)
    assert gamma(x).is_positive is True
    assert gamma(y).is_positive is None
    assert gamma(z).is_positive is None
    assert gamma(p).is_positive is None
    assert gamma(q).is_positive is None
    assert gamma(r).is_positive is None
    assert gamma(e + S.Half).is_positive is True
    assert gamma(e - S.Half).is_positive is False