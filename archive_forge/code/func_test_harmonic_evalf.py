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
def test_harmonic_evalf():
    assert str(harmonic(1.5).evalf(n=10)) == '1.280372306'
    assert str(harmonic(1.5, 2).evalf(n=10)) == '1.154576311'
    assert str(harmonic(4.0, -3).evalf(n=10)) == '100.0000000'
    assert str(harmonic(7.0, 1.0).evalf(n=10)) == '2.592857143'
    assert str(harmonic(1, pi).evalf(n=10)) == '1.000000000'
    assert str(harmonic(2, pi).evalf(n=10)) == '1.113314732'
    assert str(harmonic(1000.0, pi).evalf(n=10)) == '1.176241563'
    assert str(harmonic(I).evalf(n=10)) == '0.6718659855 + 1.076674047*I'
    assert str(harmonic(I, I).evalf(n=10)) == '-0.3970915266 + 1.9629689*I'
    assert harmonic(-1.0, 1).evalf() is S.NaN
    assert harmonic(-2.0, 2.0).evalf() is S.NaN