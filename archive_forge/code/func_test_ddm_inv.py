from sympy.testing.pytest import raises
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import (
from sympy.polys.matrices.exceptions import (
def test_ddm_inv():
    A = []
    Ainv = []
    ddm_iinv(Ainv, A, QQ)
    assert Ainv == A
    A = []
    Ainv = []
    raises(ValueError, lambda: ddm_iinv(Ainv, A, ZZ))
    A = [[QQ(1), QQ(2)]]
    Ainv = [[QQ(0), QQ(0)]]
    raises(DMNonSquareMatrixError, lambda: ddm_iinv(Ainv, A, QQ))
    A = [[QQ(1, 1), QQ(2, 1)], [QQ(3, 1), QQ(4, 1)]]
    Ainv = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    Ainv_expected = [[QQ(-2, 1), QQ(1, 1)], [QQ(3, 2), QQ(-1, 2)]]
    ddm_iinv(Ainv, A, QQ)
    assert Ainv == Ainv_expected
    A = [[QQ(1, 1), QQ(2, 1)], [QQ(2, 1), QQ(4, 1)]]
    Ainv = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    raises(DMNonInvertibleMatrixError, lambda: ddm_iinv(Ainv, A, QQ))