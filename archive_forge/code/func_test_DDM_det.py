from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_det():
    A = DDM([], (0, 0), ZZ)
    assert A.det() == ZZ(1)
    A = DDM([[ZZ(2)]], (1, 1), ZZ)
    assert A.det() == ZZ(2)
    A = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A.det() == ZZ(-2)
    A = DDM([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(1), ZZ(2), ZZ(4)], [ZZ(1), ZZ(2), ZZ(5)]], (3, 3), ZZ)
    assert A.det() == ZZ(0)
    A = DDM([[QQ(1, 2), QQ(1, 2)], [QQ(1, 3), QQ(1, 4)]], (2, 2), QQ)
    assert A.det() == QQ(-1, 24)
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    raises(DMShapeError, lambda: A.det())
    A = DDM([], (0, 1), ZZ)
    raises(DMShapeError, lambda: A.det())