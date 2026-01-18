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
def getComponentByPosition(self, idx, default=noValue, instantiate=True):
    __doc__ = Set.__doc__
    if self._currentIdx is None or self._currentIdx != idx:
        return Set.getComponentByPosition(self, idx, default=default, instantiate=instantiate)
    return self._componentValues[idx]