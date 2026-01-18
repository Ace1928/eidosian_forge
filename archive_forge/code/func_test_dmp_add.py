from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_add():
    assert dmp_add([ZZ(1), ZZ(2)], [ZZ(1)], 0, ZZ) == dup_add([ZZ(1), ZZ(2)], [ZZ(1)], ZZ)
    assert dmp_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], 0, QQ) == dup_add([QQ(1, 2), QQ(2, 3)], [QQ(1)], QQ)
    assert dmp_add([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_add([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_add([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(1)]]]
    assert dmp_add([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(3)]]]
    assert dmp_add([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(3)]]]
    assert dmp_add([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_add([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_add([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[QQ(1, 2)]]]
    assert dmp_add([[[QQ(2, 7)]]], [[[QQ(1, 7)]]], 2, QQ) == [[[QQ(3, 7)]]]
    assert dmp_add([[[QQ(1, 7)]]], [[[QQ(2, 7)]]], 2, QQ) == [[[QQ(3, 7)]]]