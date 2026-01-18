from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_sub():
    assert dmp_sub([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == dup_sub([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    assert dmp_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == dup_sub([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)
    assert dmp_sub([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_sub([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_sub([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(-1)]]]
    assert dmp_sub([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_sub([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(-1)]]]
    assert dmp_sub([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_sub([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_sub([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[QQ(-1, 2)]]]
    assert dmp_sub([[[QQ(2, 7)]]], [[[QQ(1, 7)]]], 2, QQ) == [[[QQ(1, 7)]]]
    assert dmp_sub([[[QQ(1, 7)]]], [[[QQ(2, 7)]]], 2, QQ) == [[[QQ(-1, 7)]]]