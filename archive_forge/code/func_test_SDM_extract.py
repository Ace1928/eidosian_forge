from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_extract():
    A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
    B = A.extract([1], [1])
    assert B == SDM({0: {0: ZZ(4)}}, (1, 1), ZZ)
    B = A.extract([1, 0], [1, 0])
    assert B == SDM({0: {0: ZZ(4), 1: ZZ(3)}, 1: {0: ZZ(2), 1: ZZ(1)}}, (2, 2), ZZ)
    B = A.extract([1, 1], [1, 1])
    assert B == SDM({0: {0: ZZ(4), 1: ZZ(4)}, 1: {0: ZZ(4), 1: ZZ(4)}}, (2, 2), ZZ)
    B = A.extract([-1], [-1])
    assert B == SDM({0: {0: ZZ(4)}}, (1, 1), ZZ)
    A = SDM({}, (2, 2), ZZ)
    B = A.extract([0, 1, 0], [0, 0])
    assert B == SDM({}, (3, 2), ZZ)
    A = SDM({0: {0: ZZ(1), 1: ZZ(2)}, 1: {0: ZZ(3), 1: ZZ(4)}}, (2, 2), ZZ)
    assert A.extract([], []) == SDM.zeros((0, 0), ZZ)
    assert A.extract([1], []) == SDM.zeros((1, 0), ZZ)
    assert A.extract([], [1]) == SDM.zeros((0, 1), ZZ)
    raises(IndexError, lambda: A.extract([2], [0]))
    raises(IndexError, lambda: A.extract([0], [2]))
    raises(IndexError, lambda: A.extract([-3], [0]))
    raises(IndexError, lambda: A.extract([0], [-3]))