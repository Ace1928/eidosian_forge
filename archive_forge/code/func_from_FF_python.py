from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_FF_python(K1, a, K0):
    """Convert ``ModularInteger(int)`` to Python's ``int``. """
    return a.to_int()