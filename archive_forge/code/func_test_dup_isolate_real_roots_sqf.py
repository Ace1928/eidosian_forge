from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
from sympy.testing.pytest import raises
def test_dup_isolate_real_roots_sqf():
    R, x = ring('x', ZZ)
    assert R.dup_isolate_real_roots_sqf(0) == []
    assert R.dup_isolate_real_roots_sqf(5) == []
    assert R.dup_isolate_real_roots_sqf(x ** 2 + x) == [(-1, -1), (0, 0)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 - x) == [(0, 0), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x ** 4 + x + 1) == []
    I = [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 - 2) == I
    assert R.dup_isolate_real_roots_sqf(-x ** 2 + 2) == I
    assert R.dup_isolate_real_roots_sqf(x - 1) == [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 - 3 * x + 2) == [(1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 3 - 6 * x ** 2 + 11 * x - 6) == [(1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(x ** 4 - 10 * x ** 3 + 35 * x ** 2 - 50 * x + 24) == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert R.dup_isolate_real_roots_sqf(x ** 5 - 15 * x ** 4 + 85 * x ** 3 - 225 * x ** 2 + 274 * x - 120) == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert R.dup_isolate_real_roots_sqf(x - 10) == [(10, 10)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 - 30 * x + 200) == [(10, 10), (20, 20)]
    assert R.dup_isolate_real_roots_sqf(x ** 3 - 60 * x ** 2 + 1100 * x - 6000) == [(10, 10), (20, 20), (30, 30)]
    assert R.dup_isolate_real_roots_sqf(x ** 4 - 100 * x ** 3 + 3500 * x ** 2 - 50000 * x + 240000) == [(10, 10), (20, 20), (30, 30), (40, 40)]
    assert R.dup_isolate_real_roots_sqf(x ** 5 - 150 * x ** 4 + 8500 * x ** 3 - 225000 * x ** 2 + 2740000 * x - 12000000) == [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
    assert R.dup_isolate_real_roots_sqf(x + 1) == [(-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 + 3 * x + 2) == [(-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x ** 3 + 6 * x ** 2 + 11 * x + 6) == [(-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x ** 4 + 10 * x ** 3 + 35 * x ** 2 + 50 * x + 24) == [(-4, -4), (-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x ** 5 + 15 * x ** 4 + 85 * x ** 3 + 225 * x ** 2 + 274 * x + 120) == [(-5, -5), (-4, -4), (-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x + 10) == [(-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 + 30 * x + 200) == [(-20, -20), (-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x ** 3 + 60 * x ** 2 + 1100 * x + 6000) == [(-30, -30), (-20, -20), (-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x ** 4 + 100 * x ** 3 + 3500 * x ** 2 + 50000 * x + 240000) == [(-40, -40), (-30, -30), (-20, -20), (-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x ** 5 + 150 * x ** 4 + 8500 * x ** 3 + 225000 * x ** 2 + 2740000 * x + 12000000) == [(-50, -50), (-40, -40), (-30, -30), (-20, -20), (-10, -10)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 - 5) == [(-3, -2), (2, 3)]
    assert R.dup_isolate_real_roots_sqf(x ** 3 - 5) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 4 - 5) == [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 5 - 5) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 6 - 5) == [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 7 - 5) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 8 - 5) == [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 9 - 5) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 2 - 1) == [(-1, -1), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x ** 3 + 2 * x ** 2 - x - 2) == [(-2, -2), (-1, -1), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x ** 4 - 5 * x ** 2 + 4) == [(-2, -2), (-1, -1), (1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 5 + 3 * x ** 4 - 5 * x ** 3 - 15 * x ** 2 + 4 * x + 12) == [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x ** 6 - 14 * x ** 4 + 49 * x ** 2 - 36) == [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(2 * x ** 7 + x ** 6 - 28 * x ** 5 - 14 * x ** 4 + 98 * x ** 3 + 49 * x ** 2 - 72 * x - 36) == [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(4 * x ** 8 - 57 * x ** 6 + 210 * x ** 4 - 193 * x ** 2 + 36) == [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (0, 1), (1, 1), (2, 2), (3, 3)]
    f = 9 * x ** 2 - 2
    assert R.dup_isolate_real_roots_sqf(f) == [(-1, 0), (0, 1)]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 10)) == [(QQ(-1, 2), QQ(-3, 7)), (QQ(3, 7), QQ(1, 2))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100)) == [(QQ(-9, 19), QQ(-8, 17)), (QQ(8, 17), QQ(9, 19))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 1000)) == [(QQ(-33, 70), QQ(-8, 17)), (QQ(8, 17), QQ(33, 70))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 10000)) == [(QQ(-33, 70), QQ(-107, 227)), (QQ(107, 227), QQ(33, 70))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100000)) == [(QQ(-305, 647), QQ(-272, 577)), (QQ(272, 577), QQ(305, 647))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 1000000)) == [(QQ(-1121, 2378), QQ(-272, 577)), (QQ(272, 577), QQ(1121, 2378))]
    f = 200100012 * x ** 5 - 700390052 * x ** 4 + 700490079 * x ** 3 - 200240054 * x ** 2 + 40017 * x - 2
    assert R.dup_isolate_real_roots_sqf(f) == [(QQ(0), QQ(1, 10002)), (QQ(1, 10002), QQ(1, 10002)), (QQ(1, 2), QQ(1, 2)), (QQ(1), QQ(1)), (QQ(2), QQ(2))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100000)) == [(QQ(1, 10003), QQ(1, 10003)), (QQ(1, 10002), QQ(1, 10002)), (QQ(1, 2), QQ(1, 2)), (QQ(1), QQ(1)), (QQ(2), QQ(2))]
    a, b, c, d = (10000090000001, 2000100003, 10000300007, 10000005000008)
    f = 20001600074001600021 * x ** 4 + 1700135866278935491773999857 * x ** 3 - 2000179008931031182161141026995283662899200197 * x ** 2 - 800027600594323913802305066986600025 * x + 100000950000540000725000008
    assert R.dup_isolate_real_roots_sqf(f) == [(-a, -a), (-1, 0), (0, 1), (d, d)]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100000000000)) == [(-QQ(a), -QQ(a)), (-QQ(1, b), -QQ(1, b)), (QQ(1, c), QQ(1, c)), (QQ(d), QQ(d))]
    (u, v), B, C, (s, t) = R.dup_isolate_real_roots_sqf(f, fast=True)
    assert u < -a < v and B == (-QQ(1), QQ(0)) and (C == (QQ(0), QQ(1))) and (s < d < t)
    assert R.dup_isolate_real_roots_sqf(f, fast=True, eps=QQ(1, 100000000000000000000000000000)) == [(-QQ(a), -QQ(a)), (-QQ(1, b), -QQ(1, b)), (QQ(1, c), QQ(1, c)), (QQ(d), QQ(d))]
    f = -10 * x ** 4 + 8 * x ** 3 + 80 * x ** 2 - 32 * x - 160
    assert R.dup_isolate_real_roots_sqf(f) == [(-2, -2), (-2, -1), (2, 2), (2, 3)]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100)) == [(-QQ(2), -QQ(2)), (-QQ(23, 14), -QQ(18, 11)), (QQ(2), QQ(2)), (QQ(39, 16), QQ(22, 9))]
    f = x - 1
    assert R.dup_isolate_real_roots_sqf(f, inf=2) == []
    assert R.dup_isolate_real_roots_sqf(f, sup=0) == []
    assert R.dup_isolate_real_roots_sqf(f) == [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(f, inf=1) == [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(f, sup=1) == [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(f, inf=1, sup=1) == [(1, 1)]
    f = x ** 2 - 2
    assert R.dup_isolate_real_roots_sqf(f, inf=QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_sqf(f, inf=QQ(7, 5)) == [(QQ(7, 5), QQ(3, 2))]
    assert R.dup_isolate_real_roots_sqf(f, sup=QQ(7, 5)) == [(-2, -1)]
    assert R.dup_isolate_real_roots_sqf(f, sup=QQ(7, 4)) == [(-2, -1), (1, QQ(3, 2))]
    assert R.dup_isolate_real_roots_sqf(f, sup=-QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_sqf(f, sup=-QQ(7, 5)) == [(-QQ(3, 2), -QQ(7, 5))]
    assert R.dup_isolate_real_roots_sqf(f, inf=-QQ(7, 5)) == [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(f, inf=-QQ(7, 4)) == [(-QQ(3, 2), -1), (1, 2)]
    I = [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(f, inf=-2) == I
    assert R.dup_isolate_real_roots_sqf(f, sup=+2) == I
    assert R.dup_isolate_real_roots_sqf(f, inf=-2, sup=2) == I
    R, x = ring('x', QQ)
    f = QQ(8, 5) * x ** 2 - QQ(87374, 3855) * x - QQ(17, 771)
    assert R.dup_isolate_real_roots_sqf(f) == [(-1, 0), (14, 15)]
    R, x = ring('x', EX)
    raises(DomainError, lambda: R.dup_isolate_real_roots_sqf(x + 3))