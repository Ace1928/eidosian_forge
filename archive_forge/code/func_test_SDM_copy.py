from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_copy():
    A = SDM({0: {0: ZZ(1)}}, (2, 2), ZZ)
    B = A.copy()
    assert A == B
    A[0][0] = ZZ(2)
    assert A != B