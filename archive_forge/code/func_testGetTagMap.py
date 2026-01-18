import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetTagMap(self):
    assert self.e.tagMap.presentTypes == {univ.OctetString.tagSet: univ.OctetString(''), univ.Integer.tagSet: univ.Integer(0)}