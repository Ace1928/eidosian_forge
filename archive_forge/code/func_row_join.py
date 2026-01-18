from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, parallel_poly_from_expr
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar
def row_join(self, other):
    dm = DomainMatrix.hstack(self._dm, other._dm)
    return self.from_dm(dm)