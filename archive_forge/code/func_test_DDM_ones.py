from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_ones():
    ddmone = DDM.ones((2, 3), QQ)
    assert list(ddmone) == [[QQ(1)] * 3] * 2
    assert ddmone.shape == (2, 3)
    assert ddmone.domain == QQ