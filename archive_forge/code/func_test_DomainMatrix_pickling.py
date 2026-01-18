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
def test_DomainMatrix_pickling():
    import pickle
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM
    dM = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert pickle.loads(pickle.dumps(dM)) == dM