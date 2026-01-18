import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetTypesNearPosition(self):
    assert self.e.getTagMapNearPosition(0).presentTypes == {univ.OctetString.tagSet: univ.OctetString('')}
    assert self.e.getTagMapNearPosition(1).presentTypes == {univ.Integer.tagSet: univ.Integer(0), univ.OctetString.tagSet: univ.OctetString('')}
    assert self.e.getTagMapNearPosition(2).presentTypes == {univ.OctetString.tagSet: univ.OctetString('')}