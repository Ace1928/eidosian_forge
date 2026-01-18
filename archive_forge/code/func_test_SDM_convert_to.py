from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_convert_to():
    A = SDM({0: {1: ZZ(1)}, 1: {0: ZZ(2), 1: ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0: {1: QQ(1)}, 1: {0: QQ(2), 1: QQ(3)}}, (2, 2), QQ)
    C = A.convert_to(QQ)
    assert C == B
    assert C.domain == QQ
    D = A.convert_to(ZZ)
    assert D == A
    assert D.domain == ZZ