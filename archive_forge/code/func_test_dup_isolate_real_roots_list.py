from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_isolate_real_roots_list():
    R, x = ring('x', ZZ)
    assert R.dup_isolate_real_roots_list([x ** 2 + x, x]) == [((-1, -1), {0: 1}), ((0, 0), {0: 1, 1: 1})]
    assert R.dup_isolate_real_roots_list([x ** 2 - x, x]) == [((0, 0), {0: 1, 1: 1}), ((1, 1), {0: 1})]
    assert R.dup_isolate_real_roots_list([x + 1, x + 2, x - 1, x + 1, x - 1, x - 1]) == [((-QQ(2), -QQ(2)), {1: 1}), ((-QQ(1), -QQ(1)), {0: 1, 3: 1}), ((QQ(1), QQ(1)), {2: 1, 4: 1, 5: 1})]
    assert R.dup_isolate_real_roots_list([x + 1, x + 2, x - 1, x + 1, x - 1, x + 2]) == [((-QQ(2), -QQ(2)), {1: 1, 5: 1}), ((-QQ(1), -QQ(1)), {0: 1, 3: 1}), ((QQ(1), QQ(1)), {2: 1, 4: 1})]
    f, g = (x ** 4 - 4 * x ** 2 + 4, x - 1)
    assert R.dup_isolate_real_roots_list([f, g], inf=QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_list([f, g], inf=QQ(7, 5)) == [((QQ(7, 5), QQ(3, 2)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], sup=QQ(7, 5)) == [((-2, -1), {0: 2}), ((1, 1), {1: 1})]
    assert R.dup_isolate_real_roots_list([f, g], sup=QQ(7, 4)) == [((-2, -1), {0: 2}), ((1, 1), {1: 1}), ((1, QQ(3, 2)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], sup=-QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_list([f, g], sup=-QQ(7, 5)) == [((-QQ(3, 2), -QQ(7, 5)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], inf=-QQ(7, 5)) == [((1, 1), {1: 1}), ((1, 2), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], inf=-QQ(7, 4)) == [((-QQ(3, 2), -1), {0: 2}), ((1, 1), {1: 1}), ((1, 2), {0: 2})]
    f, g = (2 * x ** 2 - 1, x ** 2 - 2)
    assert R.dup_isolate_real_roots_list([f, g]) == [((-QQ(2), -QQ(1)), {1: 1}), ((-QQ(1), QQ(0)), {0: 1}), ((QQ(0), QQ(1)), {0: 1}), ((QQ(1), QQ(2)), {1: 1})]
    assert R.dup_isolate_real_roots_list([f, g], strict=True) == [((-QQ(3, 2), -QQ(4, 3)), {1: 1}), ((-QQ(1), -QQ(2, 3)), {0: 1}), ((QQ(2, 3), QQ(1)), {0: 1}), ((QQ(4, 3), QQ(3, 2)), {1: 1})]
    f, g = (x ** 2 - 2, x ** 3 - x ** 2 - 2 * x + 2)
    assert R.dup_isolate_real_roots_list([f, g]) == [((-QQ(2), -QQ(1)), {1: 1, 0: 1}), ((QQ(1), QQ(1)), {1: 1}), ((QQ(1), QQ(2)), {1: 1, 0: 1})]
    f, g = (x ** 3 - 2 * x, x ** 5 - x ** 4 - 2 * x ** 3 + 2 * x ** 2)
    assert R.dup_isolate_real_roots_list([f, g]) == [((-QQ(2), -QQ(1)), {1: 1, 0: 1}), ((QQ(0), QQ(0)), {0: 1, 1: 2}), ((QQ(1), QQ(1)), {1: 1}), ((QQ(1), QQ(2)), {1: 1, 0: 1})]
    f, g = (x ** 9 - 3 * x ** 8 - x ** 7 + 11 * x ** 6 - 8 * x ** 5 - 8 * x ** 4 + 12 * x ** 3 - 4 * x ** 2, x ** 5 - 2 * x ** 4 + 3 * x ** 3 - 4 * x ** 2 + 2 * x)
    assert R.dup_isolate_real_roots_list([f, g], basis=False) == [((-2, -1), {0: 2}), ((0, 0), {0: 2, 1: 1}), ((1, 1), {0: 3, 1: 2}), ((1, 2), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], basis=True) == [((-2, -1), {0: 2}, [1, 0, -2]), ((0, 0), {0: 2, 1: 1}, [1, 0]), ((1, 1), {0: 3, 1: 2}, [1, -1]), ((1, 2), {0: 2}, [1, 0, -2])]
    R, x = ring('x', EX)
    raises(DomainError, lambda: R.dup_isolate_real_roots_list([x + 3]))