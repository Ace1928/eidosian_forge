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
def testBin4(self):
    binEncBase, encoder.typeMap[univ.Real.typeId].binEncBase = (encoder.typeMap[univ.Real.typeId].binEncBase, None)
    assert encoder.encode(univ.Real((1, 2, 0))) == ints2octs((9, 3, 128, 0, 1))
    encoder.typeMap[univ.Real.typeId].binEncBase = binEncBase