import math
import sys
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import integer
from pyasn1.compat import octets
from pyasn1.type import base
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import tagmap
def reset(self):
    """Remove all components and become a |ASN.1| schema object.

        See :meth:`isValue` property for more information on the
        distinction between value and schema objects.
        """
    self._componentValues = noValue
    self._dynamicNames = self.DynamicNames()
    return self