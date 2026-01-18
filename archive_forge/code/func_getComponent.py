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
def getComponent(self, innerFlag=False):
    """Return currently assigned component of the |ASN.1| object.

        Returns
        -------
        : :py:class:`~pyasn1.type.base.PyAsn1Item`
            a PyASN1 object
        """
    if self._currentIdx is None:
        raise error.PyAsn1Error('Component not chosen')
    else:
        c = self._componentValues[self._currentIdx]
        if innerFlag and isinstance(c, Choice):
            return c.getComponent(innerFlag)
        else:
            return c