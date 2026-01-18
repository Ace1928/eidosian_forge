import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetItem(self):
    assert self.e[0] == namedtype.NamedType('first-name', univ.OctetString(''))