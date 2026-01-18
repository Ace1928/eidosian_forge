import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import to_bytes
from pyasn1.compat.octets import (int2oct, oct2int, ints2octs, null,
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful
def encodeValue(self, value, asn1Spec, encodeFun, **options):
    if asn1Spec is None:
        value = value.asOctets()
    elif not isOctetsType(value):
        value = asn1Spec.clone(value).asOctets()
    return (value, not options.get('defMode', True), True)