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
def testImpossible2(self):
    try:
        encoder.encode(univ.ObjectIdentifier((0,)))
    except PyAsn1Error:
        pass
    else:
        assert 0, 'single arc OID tolerated'