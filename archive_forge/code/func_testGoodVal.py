import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
def testGoodVal(self):
    assert self.c1.isSuperTypeOf(self.c2), 'isSuperTypeOf failed'
    assert not self.c1.isSubTypeOf(self.c2), 'isSubTypeOf failed'