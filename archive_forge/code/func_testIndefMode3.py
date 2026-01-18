import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.type import useful
from pyasn1.codec.cer import encoder
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testIndefMode3(self):
    self.s.clear()
    self.s.append('b')
    self.s.append('a')
    assert encoder.encode(self.s) == ints2octs((49, 128, 4, 1, 97, 4, 1, 98, 0, 0))