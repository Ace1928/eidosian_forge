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
def test_fibonacci():
    assert [fibonacci(n) for n in range(-3, 5)] == [2, -1, 1, 0, 1, 1, 2, 3]
    assert fibonacci(100) == 354224848179261915075
    assert [lucas(n) for n in range(-3, 5)] == [-4, 3, -1, 2, 1, 3, 4, 7]
    assert lucas(100) == 792070839848372253127
    assert fibonacci(1, x) == 1
    assert fibonacci(2, x) == x
    assert fibonacci(3, x) == x ** 2 + 1
    assert fibonacci(4, x) == x ** 3 + 2 * x
    n = Dummy('n')
    assert fibonacci(n).limit(n, S.Infinity) is S.Infinity
    assert lucas(n).limit(n, S.Infinity) is S.Infinity
    assert fibonacci(n).rewrite(sqrt) == 2 ** (-n) * sqrt(5) * ((1 + sqrt(5)) ** n - (-sqrt(5) + 1) ** n) / 5
    assert fibonacci(n).rewrite(sqrt).subs(n, 10).expand() == fibonacci(10)
    assert fibonacci(n).rewrite(GoldenRatio).subs(n, 10).evalf() == Float(fibonacci(10))
    assert lucas(n).rewrite(sqrt) == (fibonacci(n - 1).rewrite(sqrt) + fibonacci(n + 1).rewrite(sqrt)).simplify()
    assert lucas(n).rewrite(sqrt).subs(n, 10).expand() == lucas(10)
    raises(ValueError, lambda: fibonacci(-3, x))