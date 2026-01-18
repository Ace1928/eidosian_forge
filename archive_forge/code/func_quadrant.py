from sympy.core.numbers import I
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.field import Field
from sympy.polys.domains.ring import Ring
def quadrant(self):
    """Return quadrant index 0-3.

        0 is included in quadrant 0.
        """
    if self.y > 0:
        return 0 if self.x > 0 else 1
    elif self.y < 0:
        return 2 if self.x < 0 else 3
    else:
        return 0 if self.x >= 0 else 2