import sys
from tests.base import BaseTestCase
from pyasn1.type import namedval
def testDict(self):
    assert set(self.e.items()) == set([('off', 0), ('on', 1)])
    assert set(self.e.keys()) == set(['off', 'on'])
    assert set(self.e) == set(['off', 'on'])
    assert set(self.e.values()) == set([0, 1])
    assert 'on' in self.e and 'off' in self.e and ('xxx' not in self.e)
    assert 0 in self.e and 1 in self.e and (2 not in self.e)