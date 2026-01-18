import sys
from tests.base import BaseTestCase
from pyasn1.type import tag
def testSequence(self):
    assert self.t1[0] == self.t2[0] and self.t1[1] == self.t2[1] and (self.t1[2] == self.t2[2]), 'tag sequence protocol fails'