from sympy.core.numbers import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom
def element_from_poly(self, f):
    """
        Produce an element of this module, representing *f* after reduction mod
        our defining minimal polynomial.

        Parameters
        ==========

        f : :py:class:`~.Poly` over :ref:`ZZ` in same var as our defining poly.

        Returns
        =======

        :py:class:`~.PowerBasisElement`

        """
    n, k = (self.n, f.degree())
    if k >= n:
        f = f % self.T
    if f == 0:
        return self.zero()
    d, c = dup_clear_denoms(f.rep.rep, QQ, convert=True)
    c = list(reversed(c))
    ell = len(c)
    z = [ZZ(0)] * (n - ell)
    col = to_col(c + z)
    return self(col, denom=d)