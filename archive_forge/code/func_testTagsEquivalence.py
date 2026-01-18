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
def testTagsEquivalence(self):
    integer = univ.Integer(2).subtype(implicitTag=tag.Tag(tag.tagClassContext, 0, 0))
    assert decoder.decode(ints2octs((159, 128, 0, 2, 1, 2)), asn1Spec=integer) == decoder.decode(ints2octs((159, 0, 2, 1, 2)), asn1Spec=integer)