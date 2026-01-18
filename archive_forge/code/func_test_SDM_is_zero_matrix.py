from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_is_zero_matrix():
    A = SDM({0: {0: QQ(1)}}, (2, 2), QQ)
    Azero = SDM.zeros((1, 2), QQ)
    assert A.is_zero_matrix() is False
    assert Azero.is_zero_matrix() is True