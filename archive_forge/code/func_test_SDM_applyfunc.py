from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_applyfunc():
    A = SDM({0: {1: ZZ(1)}}, (2, 2), ZZ)
    B = SDM({0: {1: ZZ(2)}}, (2, 2), ZZ)
    assert A.applyfunc(lambda x: 2 * x, ZZ) == B