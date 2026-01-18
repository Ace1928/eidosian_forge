from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_QQ_python(K1, a, K0):
    """Convert a Python ``Fraction`` object to ``dtype``. """
    return GMPYRational(a.numerator, a.denominator)