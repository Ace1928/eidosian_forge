from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_diag():
    A = SDM.diag([ZZ(1), ZZ(2)], ZZ, (2, 3))
    assert A == SDM({0: {0: ZZ(1)}, 1: {1: ZZ(2)}}, (2, 3), ZZ)