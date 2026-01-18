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
def testLarge1(self):
    assert encoder.encode(univ.ObjectIdentifier((2, 18446744073709551535184467440737095))) == ints2octs((6, 17, 131, 198, 223, 212, 204, 179, 255, 255, 254, 240, 184, 214, 184, 203, 226, 183, 23))