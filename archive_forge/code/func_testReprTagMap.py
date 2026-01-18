import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testReprTagMap(self):
    assert 'TagMap' in repr(self.e.tagMap)
    assert 'OctetString' in repr(self.e.tagMap)
    assert 'Integer' in repr(self.e.tagMap)