from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
@property
def max_denom(self):
    """Return the largest denominator occurring in either endpoint. """
    return max(self.ax.denominator, self.bx.denominator, self.ay.denominator, self.by.denominator)