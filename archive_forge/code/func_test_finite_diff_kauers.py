from sympy.series.kauers import finite_diff
from sympy.series.kauers import finite_diff_kauers
from sympy.abc import x, y, z, m, n, w
from sympy.core.numbers import pi
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.concrete.summations import Sum
def test_finite_diff_kauers():
    assert finite_diff_kauers(Sum(x ** 2, (x, 1, n))) == (n + 1) ** 2
    assert finite_diff_kauers(Sum(y, (y, 1, m))) == m + 1
    assert finite_diff_kauers(Sum(x * y, (x, 1, m), (y, 1, n))) == (m + 1) * (n + 1)
    assert finite_diff_kauers(Sum(x * y ** 2, (x, 1, m), (y, 1, n))) == (n + 1) ** 2 * (m + 1)