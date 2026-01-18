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
def testPosLong(self):
    assert encoder.encode(18446744073709551615, asn1Spec=univ.Integer()) == ints2octs((2, 9, 0, 255, 255, 255, 255, 255, 255, 255, 255))