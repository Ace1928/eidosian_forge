from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_particular():
    A = DDM([[QQ(1), QQ(0)]], (1, 2), QQ)
    assert A.particular() == DDM.zeros((1, 1), QQ)