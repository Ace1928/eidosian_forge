from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import (decompose_power, Factors, Term, _gcd_terms,
from sympy.core.mul import _keep_coeff as _keep_coeff
from sympy.simplify.cse_opts import sub_pre
from sympy.testing.pytest import raises
from sympy.abc import a, b, t, x, y, z
def test_monotonic_sign():
    F = _monotonic_sign
    x = symbols('x')
    assert F(x) is None
    assert F(-x) is None
    assert F(Dummy(prime=True)) == 2
    assert F(Dummy(prime=True, odd=True)) == 3
    assert F(Dummy(composite=True)) == 4
    assert F(Dummy(composite=True, odd=True)) == 9
    assert F(Dummy(positive=True, integer=True)) == 1
    assert F(Dummy(positive=True, even=True)) == 2
    assert F(Dummy(positive=True, even=True, prime=False)) == 4
    assert F(Dummy(negative=True, integer=True)) == -1
    assert F(Dummy(negative=True, even=True)) == -2
    assert F(Dummy(zero=True)) == 0
    assert F(Dummy(nonnegative=True)) == 0
    assert F(Dummy(nonpositive=True)) == 0
    assert F(Dummy(positive=True) + 1).is_positive
    assert F(Dummy(positive=True, integer=True) - 1).is_nonnegative
    assert F(Dummy(positive=True) - 1) is None
    assert F(Dummy(negative=True) + 1) is None
    assert F(Dummy(negative=True, integer=True) - 1).is_nonpositive
    assert F(Dummy(negative=True) - 1).is_negative
    assert F(-Dummy(positive=True) + 1) is None
    assert F(-Dummy(positive=True, integer=True) - 1).is_negative
    assert F(-Dummy(positive=True) - 1).is_negative
    assert F(-Dummy(negative=True) + 1).is_positive
    assert F(-Dummy(negative=True, integer=True) - 1).is_nonnegative
    assert F(-Dummy(negative=True) - 1) is None
    x = Dummy(negative=True)
    assert F(x ** 3).is_nonpositive
    assert F(x ** 3 + log(2) * x - 1).is_negative
    x = Dummy(positive=True)
    assert F(-x ** 3).is_nonpositive
    p = Dummy(positive=True)
    assert F(1 / p).is_positive
    assert F(p / (p + 1)).is_positive
    p = Dummy(nonnegative=True)
    assert F(p / (p + 1)).is_nonnegative
    p = Dummy(positive=True)
    assert F(-1 / p).is_negative
    p = Dummy(nonpositive=True)
    assert F(p / (-p + 1)).is_nonpositive
    p = Dummy(positive=True, integer=True)
    q = Dummy(positive=True, integer=True)
    assert F(-2 / p / q).is_negative
    assert F(-2 / (p - 1) / q) is None
    assert F((p - 1) * q + 1).is_positive
    assert F(-(p - 1) * q - 1).is_negative