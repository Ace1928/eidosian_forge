from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_mul():
    A = DDM([[ZZ(1)]], (1, 1), ZZ)
    A2 = DDM([[ZZ(2)]], (1, 1), ZZ)
    assert A * ZZ(2) == A2
    assert ZZ(2) * A == A2
    raises(TypeError, lambda: [[1]] * A)
    raises(TypeError, lambda: A * [[1]])