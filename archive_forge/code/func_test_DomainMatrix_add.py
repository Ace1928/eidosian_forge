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
def test_DomainMatrix_add():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(2), ZZ(4)], [ZZ(6), ZZ(8)]], (2, 2), ZZ)
    assert A + A == A.add(A) == B
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    L = [[2, 3], [3, 4]]
    raises(TypeError, lambda: A + L)
    raises(TypeError, lambda: L + A)
    A1 = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A2 = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    raises(DMShapeError, lambda: A1 + A2)
    raises(DMShapeError, lambda: A2 + A1)
    raises(DMShapeError, lambda: A1.add(A2))
    raises(DMShapeError, lambda: A2.add(A1))
    Az = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Aq = DomainMatrix([[QQ(1), QQ(2)], [QQ(3), QQ(4)]], (2, 2), QQ)
    Asum = DomainMatrix([[QQ(2), QQ(4)], [QQ(6), QQ(8)]], (2, 2), QQ)
    assert Az + Aq == Asum
    assert Aq + Az == Asum
    raises(DMDomainError, lambda: Az.add(Aq))
    raises(DMDomainError, lambda: Aq.add(Az))
    As = DomainMatrix({0: {1: ZZ(1)}, 1: {0: ZZ(2)}}, (2, 2), ZZ)
    Ad = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    Asd = As + Ad
    Ads = Ad + As
    assert Asd == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Asd.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Ads == DomainMatrix([[1, 3], [5, 4]], (2, 2), ZZ)
    assert Ads.rep == DDM([[1, 3], [5, 4]], (2, 2), ZZ)
    raises(DMFormatError, lambda: As.add(Ad))