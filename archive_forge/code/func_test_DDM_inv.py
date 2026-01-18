from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_inv():
    A = DDM([[QQ(1, 1), QQ(2, 1)], [QQ(3, 1), QQ(4, 1)]], (2, 2), QQ)
    Ainv = DDM([[QQ(-2, 1), QQ(1, 1)], [QQ(3, 2), QQ(-1, 2)]], (2, 2), QQ)
    assert A.inv() == Ainv
    A = DDM([[QQ(1), QQ(2)]], (1, 2), QQ)
    raises(DMShapeError, lambda: A.inv())
    A = DDM([[ZZ(2)]], (1, 1), ZZ)
    raises(ValueError, lambda: A.inv())
    A = DDM([], (0, 0), QQ)
    assert A.inv() == A
    A = DDM([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    raises(DMNonInvertibleMatrixError, lambda: A.inv())