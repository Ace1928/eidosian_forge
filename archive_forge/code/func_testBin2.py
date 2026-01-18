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
def testBin2(self):
    r = univ.Real((3.25, 2, 0))
    r.binEncBase = 8
    assert encoder.encode(r) == ints2octs((9, 3, 148, 255, 13))