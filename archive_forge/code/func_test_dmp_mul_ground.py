from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.polyerrors import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import FF, ZZ, QQ
from sympy.testing.pytest import raises
def test_dmp_mul_ground():
    assert dmp_mul_ground(f_0, ZZ(2), 2, ZZ) == [[[ZZ(2), ZZ(4), ZZ(6)], [ZZ(4)]], [[ZZ(6)]], [[ZZ(8), ZZ(10), ZZ(12)], [ZZ(2), ZZ(4), ZZ(2)], [ZZ(2)]]]
    assert dmp_mul_ground(F_0, QQ(1, 2), 2, QQ) == [[[QQ(1, 14), QQ(2, 14), QQ(3, 14)], [QQ(2, 14)]], [[QQ(3, 14)]], [[QQ(4, 14), QQ(5, 14), QQ(6, 14)], [QQ(1, 14), QQ(2, 14), QQ(1, 14)], [QQ(1, 14)]]]