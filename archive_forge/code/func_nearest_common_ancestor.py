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
def nearest_common_ancestor(self, other):
    """
        Locate the nearest common ancestor of this module and another.

        Returns
        =======

        :py:class:`~.Module`, ``None``

        See Also
        ========

        Module

        """
    sA = self.ancestors(include_self=True)
    oA = other.ancestors(include_self=True)
    nca = None
    for sa, oa in zip(sA, oA):
        if sa == oa:
            nca = sa
        else:
            break
    return nca