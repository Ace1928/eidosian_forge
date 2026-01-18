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
@staticmethod
def fromBinaryString(value):
    """Create a |ASN.1| object initialized from a string of '0' and '1'.

        Parameters
        ----------
        value: :class:`str`
            Text string like '1010111'
        """
    bitNo = 8
    byte = 0
    r = []
    for v in value:
        if bitNo:
            bitNo -= 1
        else:
            bitNo = 7
            r.append(byte)
            byte = 0
        if v in ('0', '1'):
            v = int(v)
        else:
            raise error.PyAsn1Error('Non-binary OCTET STRING initializer %s' % (v,))
        byte |= v << bitNo
    r.append(byte)
    return octets.ints2octs(r)