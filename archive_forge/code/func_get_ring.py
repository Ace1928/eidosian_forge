from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def get_ring(self):
    """Returns ring associated with ``self``. """
    from sympy.polys.domains import GMPYIntegerRing
    return GMPYIntegerRing()