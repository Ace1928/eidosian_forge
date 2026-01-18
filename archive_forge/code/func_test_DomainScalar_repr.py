from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_repr():
    A = DomainScalar(ZZ(1), ZZ)
    assert repr(A) in {'1', 'mpz(1)'}