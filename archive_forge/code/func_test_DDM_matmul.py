from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_matmul():
    A = DDM([[ZZ(1)], [ZZ(2)]], (2, 1), ZZ)
    B = DDM([[ZZ(3), ZZ(4)]], (1, 2), ZZ)
    AB = DDM([[ZZ(3), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    BA = DDM([[ZZ(11)]], (1, 1), ZZ)
    assert A @ B == A.matmul(B) == AB
    assert B @ A == B.matmul(A) == BA
    raises(TypeError, lambda: A @ 1)
    raises(TypeError, lambda: A @ [[3, 4]])
    Bq = DDM([[QQ(3), QQ(4)]], (1, 2), QQ)
    raises(DMDomainError, lambda: A @ Bq)
    raises(DMDomainError, lambda: Bq @ A)
    C = DDM([[ZZ(1)]], (1, 1), ZZ)
    assert A @ C == A.matmul(C) == A
    raises(DMShapeError, lambda: C @ A)
    raises(DMShapeError, lambda: C.matmul(A))
    Z04 = DDM([], (0, 4), ZZ)
    Z40 = DDM([[]] * 4, (4, 0), ZZ)
    Z50 = DDM([[]] * 5, (5, 0), ZZ)
    Z05 = DDM([], (0, 5), ZZ)
    Z45 = DDM([[0] * 5] * 4, (4, 5), ZZ)
    Z54 = DDM([[0] * 4] * 5, (5, 4), ZZ)
    Z00 = DDM([], (0, 0), ZZ)
    assert Z04 @ Z45 == Z04.matmul(Z45) == Z05
    assert Z45 @ Z50 == Z45.matmul(Z50) == Z40
    assert Z00 @ Z04 == Z00.matmul(Z04) == Z04
    assert Z50 @ Z00 == Z50.matmul(Z00) == Z50
    assert Z00 @ Z00 == Z00.matmul(Z00) == Z00
    assert Z50 @ Z04 == Z50.matmul(Z04) == Z54
    raises(DMShapeError, lambda: Z05 @ Z40)
    raises(DMShapeError, lambda: Z05.matmul(Z40))