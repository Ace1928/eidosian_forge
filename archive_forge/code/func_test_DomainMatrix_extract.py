from sympy.testing.pytest import raises
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.functions import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.domains import FF, ZZ, QQ, EXRAW
from sympy.polys.matrices.domainmatrix import DomainMatrix, DomainScalar, DM
from sympy.polys.matrices.exceptions import (
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM
def test_DomainMatrix_extract():
    dM1 = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    dM2 = DomainMatrix([[ZZ(1), ZZ(3)], [ZZ(7), ZZ(9)]], (2, 2), ZZ)
    assert dM1.extract([0, 2], [0, 2]) == dM2
    assert dM1.to_sparse().extract([0, 2], [0, 2]) == dM2.to_sparse()
    assert dM1.extract([0, -1], [0, -1]) == dM2
    assert dM1.to_sparse().extract([0, -1], [0, -1]) == dM2.to_sparse()
    dM3 = DomainMatrix([[ZZ(1), ZZ(2), ZZ(2)], [ZZ(4), ZZ(5), ZZ(5)], [ZZ(4), ZZ(5), ZZ(5)]], (3, 3), ZZ)
    assert dM1.extract([0, 1, 1], [0, 1, 1]) == dM3
    assert dM1.to_sparse().extract([0, 1, 1], [0, 1, 1]) == dM3.to_sparse()
    empty = [([], [], (0, 0)), ([1], [], (1, 0)), ([], [1], (0, 1))]
    for rows, cols, size in empty:
        assert dM1.extract(rows, cols) == DomainMatrix.zeros(size, ZZ).to_dense()
        assert dM1.to_sparse().extract(rows, cols) == DomainMatrix.zeros(size, ZZ)
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    bad_indices = [([2], [0]), ([0], [2]), ([-3], [0]), ([0], [-3])]
    for rows, cols in bad_indices:
        raises(IndexError, lambda: dM.extract(rows, cols))
        raises(IndexError, lambda: dM.to_sparse().extract(rows, cols))