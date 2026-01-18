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
def testBin3(self):
    binEncBase, encoder.typeMap[univ.Real.typeId].binEncBase = (encoder.typeMap[univ.Real.typeId].binEncBase, 16)
    assert encoder.encode(univ.Real((0.00390625, 2, 0))) == ints2octs((9, 3, 160, 254, 1))
    encoder.typeMap[univ.Real.typeId].binEncBase = binEncBase