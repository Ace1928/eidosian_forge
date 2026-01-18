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
def testWithSchema(self):
    s = univ.SequenceOf().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))
    s2, r = decoder.decode(ints2octs((163, 15, 48, 13, 4, 11, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110)), asn1Spec=s)
    assert not r
    assert s2 == [str2octs('quick brown')]
    assert s.tagSet == s2.tagSet