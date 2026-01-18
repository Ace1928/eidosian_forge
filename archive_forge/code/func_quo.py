from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def quo(self, a, b):
    """Quotient of ``a`` and ``b``, implies ``__truediv__``. """
    return GMPYRational(a) / GMPYRational(b)