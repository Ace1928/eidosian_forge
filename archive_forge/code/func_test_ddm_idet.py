from sympy.testing.pytest import raises
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import (
from sympy.polys.matrices.exceptions import (
def test_ddm_idet():
    A = []
    assert ddm_idet(A, ZZ) == ZZ(1)
    A = [[ZZ(2)]]
    assert ddm_idet(A, ZZ) == ZZ(2)
    A = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    assert ddm_idet(A, ZZ) == ZZ(-2)
    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(3), ZZ(5)]]
    assert ddm_idet(A, ZZ) == ZZ(-1)
    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]]
    assert ddm_idet(A, ZZ) == ZZ(0)
    A = [[QQ(1, 2), QQ(1, 2)], [QQ(1, 3), QQ(1, 4)]]
    assert ddm_idet(A, QQ) == QQ(-1, 24)