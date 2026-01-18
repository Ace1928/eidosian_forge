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
def test_DomainMatrix_unify_eq():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B1 = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    B2 = DomainMatrix([[QQ(1), QQ(3)], [QQ(3), QQ(4)]], (2, 2), QQ)
    B3 = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    assert A.unify_eq(B1) is True
    assert A.unify_eq(B2) is False
    assert A.unify_eq(B3) is False