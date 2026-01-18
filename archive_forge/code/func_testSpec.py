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
def testSpec(self):
    try:
        decoder.decode(ints2octs((2, 1, 12)), asn1Spec=univ.Null()) == (12, null)
    except PyAsn1Error:
        pass
    else:
        assert 0, 'wrong asn1Spec worked out'
    assert decoder.decode(ints2octs((2, 1, 12)), asn1Spec=univ.Integer()) == (12, null)