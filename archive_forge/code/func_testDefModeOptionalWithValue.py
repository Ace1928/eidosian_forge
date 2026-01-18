import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
def testDefModeOptionalWithValue(self):
    s = self.__initOptionalWithValue()
    assert encoder.encode(s) == ints2octs((48, 8, 48, 6, 4, 4, 116, 101, 115, 116))