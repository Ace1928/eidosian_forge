from sympy.testing.pytest import raises
from sympy.core.symbol import S
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.domainscalar import DomainScalar
from sympy.polys.matrices.domainmatrix import DomainMatrix
def test_DomainScalar_to_sympy():
    B = DomainScalar(ZZ(1), ZZ)
    expr = B.to_sympy()
    assert expr.is_Integer and expr == 1