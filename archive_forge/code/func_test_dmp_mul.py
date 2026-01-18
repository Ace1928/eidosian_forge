from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_mul():
    assert dmp_mul([ZZ(5)], [ZZ(7)], 0, ZZ) == dup_mul([ZZ(5)], [ZZ(7)], ZZ)
    assert dmp_mul([QQ(5, 7)], [QQ(3, 7)], 0, QQ) == dup_mul([QQ(5, 7)], [QQ(3, 7)], QQ)
    assert dmp_mul([[[]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(1)]]], [[[]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[]]], [[[ZZ(1)]]], 2, ZZ) == [[[]]]
    assert dmp_mul([[[ZZ(2)]]], [[[ZZ(1)]]], 2, ZZ) == [[[ZZ(2)]]]
    assert dmp_mul([[[ZZ(1)]]], [[[ZZ(2)]]], 2, ZZ) == [[[ZZ(2)]]]
    assert dmp_mul([[[]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(1, 2)]]], [[[]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[]]], [[[QQ(1, 2)]]], 2, QQ) == [[[]]]
    assert dmp_mul([[[QQ(2, 7)]]], [[[QQ(1, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]
    assert dmp_mul([[[QQ(1, 7)]]], [[[QQ(2, 3)]]], 2, QQ) == [[[QQ(2, 21)]]]
    K = FF(6)
    assert dmp_mul([[K(2)], [K(1)]], [[K(3)], [K(4)]], 1, K) == [[K(5)], [K(4)]]