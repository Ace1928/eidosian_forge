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
def testExtraTrue(self):
    assert decoder.decode(ints2octs((1, 1, 1, 0, 120, 50, 50))) == (1, ints2octs((0, 120, 50, 50)))