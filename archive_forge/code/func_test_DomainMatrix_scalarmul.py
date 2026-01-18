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
def test_DomainMatrix_scalarmul():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    lamda = DomainScalar(QQ(3) / QQ(2), QQ)
    assert A * lamda == DomainMatrix([[QQ(3, 2), QQ(3)], [QQ(9, 2), QQ(6)]], (2, 2), QQ)
    assert A * 2 == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert 2 * A == DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert A * DomainScalar(ZZ(0), ZZ) == DomainMatrix({}, (2, 2), ZZ)
    assert A * DomainScalar(ZZ(1), ZZ) == A
    raises(TypeError, lambda: A * 1.5)