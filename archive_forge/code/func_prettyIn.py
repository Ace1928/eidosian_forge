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
def prettyIn(self, value):
    if isinstance(value, bytes):
        return value
    elif isinstance(value, str):
        try:
            return value.encode(self.encoding)
        except UnicodeEncodeError:
            exc = sys.exc_info()[1]
            raise error.PyAsn1UnicodeEncodeError("Can't encode string '%s' with '%s' codec" % (value, self.encoding), exc)
    elif isinstance(value, OctetString):
        return value.asOctets()
    elif isinstance(value, base.SimpleAsn1Type):
        return self.prettyIn(str(value))
    elif isinstance(value, (tuple, list)):
        return self.prettyIn(bytes(value))
    else:
        return bytes(value)