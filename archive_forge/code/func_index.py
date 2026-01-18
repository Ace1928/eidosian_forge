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
def index(self, value, start=0, stop=None):
    if stop is None:
        stop = len(self)
    indices, values = zip(*self._componentValues.items())
    values = list(values)
    try:
        return indices[values.index(value, start, stop)]
    except error.PyAsn1Error:
        raise ValueError(sys.exc_info()[1])