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
def testLeading0x80Case3(self):
    try:
        decoder.decode(ints2octs((6, 2, 128, 1)))
    except PyAsn1Error:
        pass
    else:
        assert 0, 'Leading 0x80 tolarated'