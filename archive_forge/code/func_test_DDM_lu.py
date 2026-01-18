from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_lu():
    A = DDM([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    L, U, swaps = A.lu()
    assert L == DDM([[QQ(1), QQ(0)], [QQ(3), QQ(1)]], (2, 2), QQ)
    assert U == DDM([[QQ(1), QQ(2)], [QQ(0), QQ(-2)]], (2, 2), QQ)
    assert swaps == []
    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 2]]
    Lexp = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
    Uexp = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    to_dom = lambda rows, dom: [[dom(e) for e in row] for row in rows]
    A = DDM(to_dom(A, QQ), (4, 4), QQ)
    Lexp = DDM(to_dom(Lexp, QQ), (4, 4), QQ)
    Uexp = DDM(to_dom(Uexp, QQ), (4, 4), QQ)
    L, U, swaps = A.lu()
    assert L == Lexp
    assert U == Uexp
    assert swaps == []