from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_particular():
    A = SDM({0: {0: QQ(1)}}, (2, 2), QQ)
    Apart = SDM.zeros((1, 2), QQ)
    assert A.particular() == Apart