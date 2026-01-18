import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
def testBinDefault(self):

    class BinDefault(univ.OctetString):
        defaultBinValue = '1000010111101110101111000000111011'
    assert BinDefault() == univ.OctetString(binValue='1000010111101110101111000000111011')