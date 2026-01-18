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
def testEdge9(self):
    assert encoder.encode(univ.ObjectIdentifier((2, 16843570))) == ints2octs((6, 4, 136, 132, 135, 2))