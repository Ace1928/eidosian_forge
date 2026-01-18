from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_FF_gmpy(K1, a, K0):
    """Convert ``ModularInteger(mpz)`` to Python's ``int``. """
    return PythonInteger(a.to_int())