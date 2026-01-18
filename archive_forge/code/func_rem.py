from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.rationalfield import RationalField
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def rem(self, a, b):
    """Remainder of ``a`` and ``b``, implies nothing.  """
    return self.zero