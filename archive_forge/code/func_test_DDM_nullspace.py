from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_nullspace():
    A = DDM([[QQ(1), QQ(1)], [QQ(1), QQ(1)]], (2, 2), QQ)
    Anull = DDM([[QQ(-1), QQ(1)]], (1, 2), QQ)
    nonpivots = [1]
    assert A.nullspace() == (Anull, nonpivots)