import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
def testBadVal(self):
    assert not self.c2.isSuperTypeOf(self.c1), 'isSuperTypeOf failed'
    assert self.c2.isSubTypeOf(self.c1), 'isSubTypeOf failed'