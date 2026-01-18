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
def test_euler_polynomials():
    assert euler(0, x) == 1
    assert euler(1, x) == x - S.Half
    assert euler(2, x) == x ** 2 - x
    assert euler(3, x) == x ** 3 - 3 * x ** 2 / 2 + Rational(1, 4)
    m = Symbol('m')
    assert isinstance(euler(m, x), euler)
    from sympy.core.numbers import Float
    A = Float('-0.46237208575048694923364757452876131e8')
    B = euler(19, S.Pi).evalf(32)
    assert abs((A - B) / A) < 1e-31