from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_GaussianRationalField(K1, a, K0):
    """Convert a ``GaussianElement`` object to ``dtype``. """
    if a.y == 0:
        return GMPYRational(a.x)