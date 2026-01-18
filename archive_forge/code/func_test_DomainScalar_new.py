from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_new():
    A = DomainScalar(ZZ(1), ZZ)
    B = A.new(ZZ(4), ZZ)
    assert B == DomainScalar(ZZ(4), ZZ)