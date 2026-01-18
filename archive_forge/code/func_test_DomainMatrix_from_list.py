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
def test_DomainMatrix_from_list():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_list([[1, 2], [3, 4]], ZZ)
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == ZZ
    dom = FF(7)
    ddm = DDM([[dom(1), dom(2)], [dom(3), dom(4)]], (2, 2), dom)
    A = DomainMatrix.from_list([[1, 2], [3, 4]], dom)
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == dom
    ddm = DDM([[QQ(1, 2), QQ(3, 1)], [QQ(1, 4), QQ(5, 1)]], (2, 2), QQ)
    A = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == QQ