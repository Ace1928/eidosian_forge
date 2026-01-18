import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import char
from pyasn1.type import univ
from pyasn1.type import constraint
from pyasn1.compat.octets import ints2octs
from pyasn1.error import PyAsn1Error
def testPrintable(self):
    if sys.version_info[0] < 3:
        assert unicode(self.asn1String) == self.pythonString, '__str__() fails'
    else:
        assert str(self.asn1String) == self.pythonString, '__str__() fails'