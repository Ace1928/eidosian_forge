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
def test_DomainMatrix_pow():
    eye = DomainMatrix.eye(2, ZZ)
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(7), ZZ(10)], [ZZ(15), ZZ(22)]], (2, 2), ZZ)
    A3 = DomainMatrix([[ZZ(37), ZZ(54)], [ZZ(81), ZZ(118)]], (2, 2), ZZ)
    assert A ** 0 == A.pow(0) == eye
    assert A ** 1 == A.pow(1) == A
    assert A ** 2 == A.pow(2) == A2
    assert A ** 3 == A.pow(3) == A3
    raises(TypeError, lambda: A ** Rational(1, 2))
    raises(NotImplementedError, lambda: A ** (-1))
    raises(NotImplementedError, lambda: A.pow(-1))
    A = DomainMatrix.zeros((2, 1), ZZ)
    raises(DMNonSquareMatrixError, lambda: A ** 1)