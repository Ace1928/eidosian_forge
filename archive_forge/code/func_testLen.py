import sys
from tests.base import BaseTestCase
from pyasn1.type import namedval
def testLen(self):
    assert len(self.e) == 2
    assert len(namedval.NamedValues()) == 0