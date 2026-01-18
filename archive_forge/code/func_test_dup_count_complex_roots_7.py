from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_count_complex_roots_7():
    R, x = ring('x', ZZ)
    f = x ** 4 + 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1
    f = x ** 5 - 2 * x ** 4 + 4 * x - 8
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1
    f = x ** 6 - 2 * x ** 4 + 4 * x ** 2 - 8
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, c, d) == 1
    f = x ** 5 - x ** 4 + 4 * x - 4
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 2
    f = x ** 6 - x ** 5 + 4 * x ** 2 - 4 * x
    assert R.dup_count_complex_roots(f, a, b) == 6
    assert R.dup_count_complex_roots(f, c, d) == 3
    f = x ** 5 + x ** 4 + 4 * x + 4
    assert R.dup_count_complex_roots(f, a, b) == 5
    assert R.dup_count_complex_roots(f, c, d) == 1
    f = x ** 6 + x ** 5 + 4 * x ** 2 + 4 * x
    assert R.dup_count_complex_roots(f, a, b) == 6
    assert R.dup_count_complex_roots(f, c, d) == 2
    f = x ** 6 - x ** 4 + 4 * x ** 2 - 4
    assert R.dup_count_complex_roots(f, a, b) == 6
    assert R.dup_count_complex_roots(f, c, d) == 2
    f = x ** 7 - x ** 5 + 4 * x ** 3 - 4 * x
    assert R.dup_count_complex_roots(f, a, b) == 7
    assert R.dup_count_complex_roots(f, c, d) == 3
    f = x ** 8 + 3 * x ** 4 - 4
    assert R.dup_count_complex_roots(f, a, b) == 8
    assert R.dup_count_complex_roots(f, c, d) == 3