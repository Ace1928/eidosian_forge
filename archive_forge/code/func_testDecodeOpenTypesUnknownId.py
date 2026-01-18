import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import univ
from pyasn1.type import char
from pyasn1.codec.ber import decoder
from pyasn1.codec.ber import eoo
from pyasn1.compat.octets import ints2octs, str2octs, null
from pyasn1.error import PyAsn1Error
def testDecodeOpenTypesUnknownId(self):
    s, r = decoder.decode(ints2octs((48, 8, 2, 1, 3, 163, 3, 2, 1, 12)), asn1Spec=self.s, decodeOpenTypes=True)
    assert not r
    assert s[0] == 3
    assert s[1] == univ.OctetString(hexValue='02010C')