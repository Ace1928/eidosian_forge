import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.codec.der import encoder
from pyasn1.compat.octets import ints2octs
def testDefModeOptionalWithDefault(self):
    s = self.__initOptionalWithDefault()
    assert encoder.encode(s) == ints2octs((48, 5, 48, 3, 2, 1, 123))