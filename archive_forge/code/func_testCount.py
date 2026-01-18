import math
import pickle
import sys
from tests.base import BaseTestCase
from pyasn1.type import univ
from pyasn1.type import tag
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import error
from pyasn1.compat.octets import str2octs, ints2octs, octs2ints
from pyasn1.error import PyAsn1Error
def testCount(self):
    self.s1.clear()
    for x in ['abc', 'def', 'abc']:
        self.s1.append(x)
    assert self.s1.count(str2octs('abc')) == 2
    assert self.s1.count(str2octs('def')) == 1
    assert self.s1.count(str2octs('ghi')) == 0