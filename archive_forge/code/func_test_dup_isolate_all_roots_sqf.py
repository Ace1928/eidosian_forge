from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_isolate_all_roots_sqf():
    R, x = ring('x', ZZ)
    f = 4 * x ** 4 - x ** 3 + 2 * x ** 2 + 5 * x
    assert R.dup_isolate_all_roots_sqf(f) == ([(-1, 0), (0, 0)], [((0, -QQ(5, 2)), (QQ(5, 2), 0)), ((0, 0), (QQ(5, 2), QQ(5, 2)))])
    assert R.dup_isolate_all_roots_sqf(f, eps=QQ(1, 10)) == ([(QQ(-7, 8), QQ(-6, 7)), (0, 0)], [((QQ(35, 64), -QQ(35, 32)), (QQ(5, 8), -QQ(65, 64))), ((QQ(35, 64), QQ(65, 64)), (QQ(5, 8), QQ(35, 32)))])