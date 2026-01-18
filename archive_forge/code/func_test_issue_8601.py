import string
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core import (EulerGamma, TribonacciConstant)
from sympy.core.numbers import (Float, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.numbers import carmichael
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.integers import floor
from sympy.polys.polytools import cancel
from sympy.series.limits import limit, Limit
from sympy.series.order import O
from sympy.functions import (
from sympy.functions.combinatorial.numbers import _nT
from sympy.core.expr import unchanged
from sympy.core.numbers import GoldenRatio, Integer
from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy
from sympy.abc import x
def test_issue_8601():
    n = Symbol('n', integer=True, negative=True)
    assert catalan(n - 1) is S.Zero
    assert catalan(Rational(-1, 2)) is S.ComplexInfinity
    assert catalan(-S.One) == Rational(-1, 2)
    c1 = catalan(-5.6).evalf()
    assert str(c1) == '6.93334070531408e-5'
    c2 = catalan(-35.4).evalf()
    assert str(c2) == '-4.14189164517449e-24'