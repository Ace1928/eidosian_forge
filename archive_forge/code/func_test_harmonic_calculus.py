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
def test_harmonic_calculus():
    y = Symbol('y', positive=True)
    z = Symbol('z', negative=True)
    assert harmonic(x, 1).limit(x, 0) == 0
    assert harmonic(x, y).limit(x, 0) == 0
    assert harmonic(x, 1).series(x, y, 2) == harmonic(y) + (x - y) * zeta(2, y + 1) + O((x - y) ** 2, (x, y))
    assert limit(harmonic(x, y), x, oo) == harmonic(oo, y)
    assert limit(harmonic(x, y + 1), x, oo) == zeta(y + 1)
    assert limit(harmonic(x, y - 1), x, oo) == harmonic(oo, y - 1)
    assert limit(harmonic(x, z), x, oo) == Limit(harmonic(x, z), x, oo, dir='-')
    assert limit(harmonic(x, z + 1), x, oo) == oo
    assert limit(harmonic(x, z + 2), x, oo) == harmonic(oo, z + 2)
    assert limit(harmonic(x, z - 1), x, oo) == Limit(harmonic(x, z - 1), x, oo, dir='-')