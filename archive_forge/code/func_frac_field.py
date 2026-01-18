from sympy.polys.agca.modules import FreeModuleQuotientRing
from sympy.polys.domains.ring import Ring
from sympy.polys.polyerrors import NotReversible, CoercionFailed
from sympy.utilities import public
def frac_field(self, *gens):
    """Returns a fraction field, i.e. ``K(X)``. """
    raise NotImplementedError('nested domains not allowed')