from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_ZZ_gmpy(K1, a, K0):
    """Convert a GMPY ``mpz`` object to ``dtype``. """
    return GMPYRational(a)