from sympy.testing.pytest import raises
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import (
from sympy.polys.matrices.exceptions import (
def test_ddm_charpoly():
    A = []
    assert ddm_berk(A, ZZ) == [[ZZ(1)]]
    A = [[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]]
    Avec = [[ZZ(1)], [ZZ(-15)], [ZZ(-18)], [ZZ(0)]]
    assert ddm_berk(A, ZZ) == Avec
    A = DDM([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: ddm_berk(A, ZZ))