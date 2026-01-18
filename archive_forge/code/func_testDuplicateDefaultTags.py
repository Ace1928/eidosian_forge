import sys
from tests.base import BaseTestCase
from pyasn1.type import namedtype
from pyasn1.type import univ
from pyasn1.error import PyAsn1Error
def testDuplicateDefaultTags(self):
    nt = namedtype.NamedTypes(namedtype.NamedType('first-name', univ.Any()), namedtype.NamedType('age', univ.Any()))
    assert isinstance(nt.tagMap, namedtype.NamedTypes.PostponedError)