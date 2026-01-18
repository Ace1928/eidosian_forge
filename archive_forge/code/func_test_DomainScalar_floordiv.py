from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_floordiv():
    A = DomainScalar(ZZ(-5), ZZ)
    B = DomainScalar(QQ(2), QQ)
    assert A // B == DomainScalar(QQ(-5, 2), QQ)
    C = DomainScalar(ZZ(2), ZZ)
    assert A // C == DomainScalar(ZZ(-3), ZZ)
    raises(TypeError, lambda: A // 1.5)