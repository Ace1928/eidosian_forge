from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_mul():
    A = DomainScalar(ZZ(1), ZZ)
    B = DomainScalar(QQ(2), QQ)
    dm = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    assert A * B == DomainScalar(QQ(2), QQ)
    assert A * dm == dm
    assert B * 2 == DomainScalar(QQ(4), QQ)
    raises(TypeError, lambda: A * 1.5)