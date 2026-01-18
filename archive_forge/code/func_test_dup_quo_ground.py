from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_quo_ground():
    raises(ZeroDivisionError, lambda: dup_quo_ground(dup_normal([1, 2, 3], ZZ), ZZ(0), ZZ))
    f = dup_normal([], ZZ)
    assert dup_quo_ground(f, ZZ(3), ZZ) == dup_normal([], ZZ)
    f = dup_normal([6, 2, 8], ZZ)
    assert dup_quo_ground(f, ZZ(1), ZZ) == f
    assert dup_quo_ground(f, ZZ(2), ZZ) == dup_normal([3, 1, 4], ZZ)
    assert dup_quo_ground(f, ZZ(3), ZZ) == dup_normal([2, 0, 2], ZZ)
    f = dup_normal([6, 2, 8], QQ)
    assert dup_quo_ground(f, QQ(1), QQ) == f
    assert dup_quo_ground(f, QQ(2), QQ) == [QQ(3), QQ(1), QQ(4)]
    assert dup_quo_ground(f, QQ(7), QQ) == [QQ(6, 7), QQ(2, 7), QQ(8, 7)]