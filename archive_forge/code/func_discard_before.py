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
def discard_before(self, r):
    """
        Produce a new module by discarding all generators before a given
        index *r*.
        """
    W = self.matrix[:, r:]
    s = self.n - r
    M = None
    mt = self._mult_tab
    if mt is not None:
        M = {}
        for u in range(s):
            M[u] = {}
            for v in range(u, s):
                M[u][v] = mt[r + u][r + v][r:]
    return Submodule(self.parent, W, denom=self.denom, mult_tab=M)