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
def test_DomainMatrix_rowspace():
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ)
    assert A.rowspace() == A
    Az = DomainMatrix([[ZZ(1), ZZ(-1), ZZ(1)], [ZZ(2), ZZ(-2), ZZ(3)]], (2, 3), ZZ)
    raises(DMNotAField, lambda: Az.rowspace())
    A = DomainMatrix([[QQ(1), QQ(-1), QQ(1)], [QQ(2), QQ(-2), QQ(3)]], (2, 3), QQ, fmt='sparse')
    assert A.rowspace() == A