from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_RealField(K1, a, K0):
    """Convert a mpmath ``mpf`` object to ``dtype``. """
    return GMPYRational(*map(int, K0.to_rational(a)))