from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_DDM_str():
    sdm = SDM({0: {0: ZZ(1)}, 1: {1: ZZ(1)}}, (2, 2), ZZ)
    assert str(sdm) == '{0: {0: 1}, 1: {1: 1}}'
    if HAS_GMPY:
        assert repr(sdm) == 'SDM({0: {0: mpz(1)}, 1: {1: mpz(1)}}, (2, 2), ZZ)'
    else:
        assert repr(sdm) == 'SDM({0: {0: 1}, 1: {1: 1}}, (2, 2), ZZ)'