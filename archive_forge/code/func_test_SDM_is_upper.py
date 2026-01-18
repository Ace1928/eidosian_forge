from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_is_upper():
    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)}, 1: {1: QQ(5), 2: QQ(6), 3: QQ(7)}, 2: {2: QQ(8), 3: QQ(9)}}, (3, 4), QQ)
    B = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3), 3: QQ(4)}, 1: {1: QQ(5), 2: QQ(6), 3: QQ(7)}, 2: {1: QQ(7), 2: QQ(8), 3: QQ(9)}}, (3, 4), QQ)
    assert A.is_upper() is True
    assert B.is_upper() is False