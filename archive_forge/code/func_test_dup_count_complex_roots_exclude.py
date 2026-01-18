from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_count_complex_roots_exclude():
    R, x = ring('x', ZZ)
    f = x ** 5 - x
    a, b = ((-QQ(1), QQ(0)), (QQ(1), QQ(1)))
    assert R.dup_count_complex_roots(f, a, b) == 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['S']) == 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['N']) == 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['S', 'N']) == 2
    assert R.dup_count_complex_roots(f, a, b, exclude=['E']) == 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['W']) == 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['E', 'W']) == 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['N', 'S', 'E', 'W']) == 2
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW']) == 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['SE']) == 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE']) == 2
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE', 'S']) == 1
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE', 'S', 'N']) == 0
    a, b = ((QQ(0), QQ(0)), (QQ(1), QQ(1)))
    assert R.dup_count_complex_roots(f, a, b, exclude=True) == 1