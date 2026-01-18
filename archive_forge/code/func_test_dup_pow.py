from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dup_pow():
    assert dup_pow([], 0, ZZ) == [ZZ(1)]
    assert dup_pow([], 0, QQ) == [QQ(1)]
    assert dup_pow([], 1, ZZ) == []
    assert dup_pow([], 7, ZZ) == []
    assert dup_pow([ZZ(1)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 1, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(1)], 7, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(3)], 0, ZZ) == [ZZ(1)]
    assert dup_pow([ZZ(3)], 1, ZZ) == [ZZ(3)]
    assert dup_pow([ZZ(3)], 7, ZZ) == [ZZ(2187)]
    assert dup_pow([QQ(1, 1)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 1, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(1, 1)], 7, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(3, 7)], 0, QQ) == [QQ(1, 1)]
    assert dup_pow([QQ(3, 7)], 1, QQ) == [QQ(3, 7)]
    assert dup_pow([QQ(3, 7)], 7, QQ) == [QQ(2187, 823543)]
    f = dup_normal([2, 0, 0, 1, 7], ZZ)
    assert dup_pow(f, 0, ZZ) == dup_normal([1], ZZ)
    assert dup_pow(f, 1, ZZ) == dup_normal([2, 0, 0, 1, 7], ZZ)
    assert dup_pow(f, 2, ZZ) == dup_normal([4, 0, 0, 4, 28, 0, 1, 14, 49], ZZ)
    assert dup_pow(f, 3, ZZ) == dup_normal([8, 0, 0, 12, 84, 0, 6, 84, 294, 1, 21, 147, 343], ZZ)