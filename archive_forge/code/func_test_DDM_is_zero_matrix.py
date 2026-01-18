from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_is_zero_matrix():
    A = DDM([[QQ(1), QQ(0)], [QQ(0), QQ(0)]], (2, 2), QQ)
    Azero = DDM.zeros((1, 2), QQ)
    assert A.is_zero_matrix() is False
    assert Azero.is_zero_matrix() is True