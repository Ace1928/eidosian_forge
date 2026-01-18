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
def testSetComponents(self):
    assert self.s1.clone().setComponents(name='a', nick='b', age=1) == self.s1.setComponentByPosition(0, 'a').setComponentByPosition(1, 'b').setComponentByPosition(2, 1)