from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_eye():
    A = SDM.eye((2, 2), ZZ)
    assert A.domain == ZZ
    assert A.shape == (2, 2)
    assert dict(A) == {0: {0: ZZ(1)}, 1: {1: ZZ(1)}}