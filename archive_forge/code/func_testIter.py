import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testIter(self):
    assert list(self.e) == ['first-name', 'age', 'family-name']