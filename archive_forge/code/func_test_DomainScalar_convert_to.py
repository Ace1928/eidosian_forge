from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_convert_to():
    A = DomainScalar(ZZ(1), ZZ)
    B = A.convert_to(QQ)
    assert B == DomainScalar(QQ(1), QQ)