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
def testSet(self):
    assert self.b.clone('Active') == (1,)
    assert self.b.clone('Urgent') == (0, 1)
    assert self.b.clone('Urgent, Active') == (1, 1)
    assert self.b.clone("'1010100110001010'B") == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
    assert self.b.clone("'A98A'H") == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
    assert self.b.clone(binValue='1010100110001010') == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
    assert self.b.clone(hexValue='A98A') == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
    assert self.b.clone('1010100110001010') == (1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0)
    assert self.b.clone((1, 0, 1)) == (1, 0, 1)