import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetPositionByName(self):
    assert self.e.getPositionByName('first-name') == 0, 'getPositionByName() fails'