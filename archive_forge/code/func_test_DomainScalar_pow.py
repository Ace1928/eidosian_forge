from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_pow():
    A = DomainScalar(ZZ(-5), ZZ)
    B = A ** 2
    assert B == DomainScalar(ZZ(25), ZZ)
    raises(TypeError, lambda: A ** 1.5)