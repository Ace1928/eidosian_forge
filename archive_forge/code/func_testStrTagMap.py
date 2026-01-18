import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testStrTagMap(self):
    assert 'TagMap' in str(self.e.tagMap)
    assert 'OctetString' in str(self.e.tagMap)
    assert 'Integer' in str(self.e.tagMap)