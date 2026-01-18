from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_mul_elementwise():
    A = SDM({0: {0: ZZ(2), 1: ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0: {0: ZZ(4)}, 1: {0: ZZ(3)}}, (2, 2), ZZ)
    C = SDM({0: {0: ZZ(8)}}, (2, 2), ZZ)
    assert A.mul_elementwise(B) == C
    assert B.mul_elementwise(A) == C
    Aq = A.convert_to(QQ)
    A1 = SDM({0: {0: ZZ(1)}}, (1, 1), ZZ)
    raises(DMDomainError, lambda: Aq.mul_elementwise(B))
    raises(DMShapeError, lambda: A1.mul_elementwise(B))