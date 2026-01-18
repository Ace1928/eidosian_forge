from sympy.polys.domains.field import Field
from sympy.polys.domains.modularinteger import ModularIntegerFactory
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
from sympy.polys.domains.groundtypes import SymPyInteger
def from_FF(K1, a, K0=None):
    """Convert ``ModularInteger(int)`` to ``dtype``. """
    return K1.dtype(K1.dom.from_ZZ(a.val, K0.dom))