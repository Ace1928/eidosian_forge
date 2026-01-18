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
def test_DomainMatrix_solve():
    A = DomainMatrix([[QQ(1), QQ(2)], [QQ(2), QQ(4)]], (2, 2), QQ)
    b = DomainMatrix([[QQ(1)], [QQ(2)]], (2, 1), QQ)
    particular = DomainMatrix([[1, 0]], (1, 2), QQ)
    nullspace = DomainMatrix([[-2, 1]], (1, 2), QQ)
    assert A._solve(b) == (particular, nullspace)
    b3 = DomainMatrix([[QQ(1)], [QQ(1)], [QQ(1)]], (3, 1), QQ)
    raises(DMShapeError, lambda: A._solve(b3))
    bz = DomainMatrix([[ZZ(1)], [ZZ(1)]], (2, 1), ZZ)
    raises(DMNotAField, lambda: A._solve(bz))