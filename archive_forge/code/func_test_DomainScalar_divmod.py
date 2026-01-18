from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_divmod():
    A = DomainScalar(ZZ(5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert divmod(A, B) == (DomainScalar(QQ(5, 2), QQ), DomainScalar(QQ(0), QQ))
    C = DomainScalar(ZZ(2), ZZ)
    assert divmod(A, C) == (DomainScalar(ZZ(2), ZZ), DomainScalar(ZZ(1), ZZ))
    raises(TypeError, lambda: divmod(A, 1.5))