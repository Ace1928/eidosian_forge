from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_convert_to():
    ddm = DDM([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    assert ddm.convert_to(ZZ) == ddm
    ddmq = ddm.convert_to(QQ)
    assert ddmq.domain == QQ