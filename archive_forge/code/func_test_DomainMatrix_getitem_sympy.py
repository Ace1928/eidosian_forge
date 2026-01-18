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
def test_DomainMatrix_getitem_sympy():
    dM = DomainMatrix({2: {2: ZZ(2)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    val1 = dM.getitem_sympy(0, 0)
    assert val1 is S.Zero
    val2 = dM.getitem_sympy(2, 2)
    assert val2 == 2 and isinstance(val2, Integer)