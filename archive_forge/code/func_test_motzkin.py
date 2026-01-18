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
def test_motzkin():
    assert motzkin.is_motzkin(4) == True
    assert motzkin.is_motzkin(9) == True
    assert motzkin.is_motzkin(10) == False
    assert motzkin.find_motzkin_numbers_in_range(10, 200) == [21, 51, 127]
    assert motzkin.find_motzkin_numbers_in_range(10, 400) == [21, 51, 127, 323]
    assert motzkin.find_motzkin_numbers_in_range(10, 1600) == [21, 51, 127, 323, 835]
    assert motzkin.find_first_n_motzkins(5) == [1, 1, 2, 4, 9]
    assert motzkin.find_first_n_motzkins(7) == [1, 1, 2, 4, 9, 21, 51]
    assert motzkin.find_first_n_motzkins(10) == [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]
    raises(ValueError, lambda: motzkin.eval(77.58))
    raises(ValueError, lambda: motzkin.eval(-8))
    raises(ValueError, lambda: motzkin.find_motzkin_numbers_in_range(-2, 7))
    raises(ValueError, lambda: motzkin.find_motzkin_numbers_in_range(13, 7))
    raises(ValueError, lambda: motzkin.find_first_n_motzkins(112.8))