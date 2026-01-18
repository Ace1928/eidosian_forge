import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testEncodeOpenTypeIncompatibleType(self):
    self.s.clear()
    self.s[0] = 2
    self.s[1] = univ.ObjectIdentifier('1.3.6')
    try:
        encoder.encode(self.s, asn1Spec=self.s)
    except PyAsn1Error:
        assert False, 'incompatible open type tolerated'