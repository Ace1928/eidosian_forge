from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_lu():
    A = SDM({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)
    L = SDM({0: {0: QQ(1)}, 1: {0: QQ(3), 1: QQ(1)}}, (2, 2), QQ)
    assert A.lu()[0] == L