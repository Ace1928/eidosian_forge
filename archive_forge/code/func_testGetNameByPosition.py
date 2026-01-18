import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetNameByPosition(self):
    assert self.e.getNameByPosition(0) == 'first-name', 'getNameByPosition() fails'