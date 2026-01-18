import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testGetTagMapWithDups(self):
    try:
        self.e.tagMapUnique[0]
    except PyAsn1Error:
        pass
    else:
        assert 0, 'Duped types not noticed'