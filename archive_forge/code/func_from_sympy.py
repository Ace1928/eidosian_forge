from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_sympy(self, a):
    """Convert SymPy's Integer to ``dtype``. """
    if a.is_Rational:
        return GMPYRational(a.p, a.q)
    elif a.is_Float:
        from sympy.polys.domains import RR
        return GMPYRational(*map(int, RR.to_rational(a)))
    else:
        raise CoercionFailed('expected ``Rational`` object, got %s' % a)