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
@classmethod
def from_int_list(cls, module, coeffs, denom=1):
    """
        Make a :py:class:`~.ModuleElement` from a list of ints (instead of a
        column vector).
        """
    col = to_col(coeffs)
    return cls(module, col, denom=denom)