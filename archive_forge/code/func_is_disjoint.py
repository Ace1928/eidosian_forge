from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def is_disjoint(self, other):
    """Return ``True`` if two isolation intervals are disjoint. """
    if isinstance(other, RealInterval):
        return other.is_disjoint(self)
    if self.conj != other.conj:
        return True
    re_distinct = self.bx < other.ax or other.bx < self.ax
    if re_distinct:
        return True
    im_distinct = self.by < other.ay or other.by < self.ay
    return im_distinct