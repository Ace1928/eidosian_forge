import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testOptionalWithDefaultAndOptional(self):
    s = self.__initOptionalWithDefaultAndOptional()
    assert encoder.encode(s) == ints2octs((48, 128, 48, 128, 4, 4, 116, 101, 115, 116, 2, 1, 123, 0, 0, 0, 0))