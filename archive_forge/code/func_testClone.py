import sys
from tests.base import BaseTestCase
from pyasn1.type import namedval
def testClone(self):
    assert namedval.NamedValues(off=0).clone(('on', 1)) == {'off': 0, 'on': 1}
    assert namedval.NamedValues(off=0).clone(on=1) == {'off': 0, 'on': 1}