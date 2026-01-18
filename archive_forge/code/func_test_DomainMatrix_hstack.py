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
def test_DomainMatrix_hstack():
    A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
    AB = DomainMatrix([[ZZ(1), ZZ(2), ZZ(5), ZZ(6)], [ZZ(3), ZZ(4), ZZ(7), ZZ(8)]], (2, 4), ZZ)
    ABC = DomainMatrix([[ZZ(1), ZZ(2), ZZ(5), ZZ(6), ZZ(9), ZZ(10)], [ZZ(3), ZZ(4), ZZ(7), ZZ(8), ZZ(11), ZZ(12)]], (2, 6), ZZ)
    assert A.hstack(B) == AB
    assert A.hstack(B, C) == ABC