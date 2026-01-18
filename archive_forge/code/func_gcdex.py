from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def gcdex(self, a, b):
    """Compute extended GCD of ``a`` and ``b``. """
    return python_gcdex(a, b)