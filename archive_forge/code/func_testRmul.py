import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import char
from pyasn1.type import univ
from pyasn1.type import constraint
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testRmul(self):
    assert 2 * self.asn1String == 2 * self.pythonString, '__rmul__() fails'